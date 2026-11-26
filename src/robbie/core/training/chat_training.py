"""Resumable, token-weighted SFT for Robbie's own transformer."""

from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import random
import time

import torch

from robbie.core.data.chat_data import collate_answers
from robbie.core.models.transformer import RobbieTransformer


@dataclass
class ChatTrainingConfig:
    epochs: int = 3
    batch_size: int = 4
    accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1
    seed: int = 42
    eval_every: int = 250
    save_every: int = 100
    log_every: int = 10
    eval_examples: int = 1000
    device: str = "cuda"
    amp: bool = True


def load_initial_model(path, context):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = replace(checkpoint["config"], block_size=context, dropout=0.05)
    model = RobbieTransformer(config)
    state = checkpoint["model_state_dict"]
    old_positions = state["wpe.weight"]
    with torch.no_grad():
        length = min(context, len(old_positions))
        model.wpe.weight[:length].copy_(old_positions[:length])
    compatible = {key: value for key, value in state.items()
                  if key != "wpe.weight" and not key.endswith(".attn.bias")}
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    allowed = {"wpe.weight"} | {f"blocks.{i}.attn.bias" for i in range(config.n_layer)}
    if set(missing) != allowed or unexpected:
        raise ValueError(f"Incompatible checkpoint: {missing}, {unexpected}")
    return model


def atomic_save(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class ChatTrainingRun:
    def __init__(self, model, splits, tokenizer, config, output_dir, data_fingerprint,
                 sample_callback=None):
        self.model, self.splits, self.tokenizer = model, splits, tokenizer
        self.config, self.output_dir = config, Path(output_dir)
        self.data_fingerprint, self.sample_callback = data_fingerprint, sample_callback
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = torch.optim.AdamW([
            {"params": [p for p in model.parameters() if p.requires_grad and p.ndim >= 2],
             "weight_decay": config.weight_decay},
            {"params": [p for p in model.parameters() if p.requires_grad and p.ndim < 2],
             "weight_decay": 0.0},
        ], lr=config.learning_rate)
        self.amp_enabled = config.amp and self.device.type == "cuda"
        self.amp_dtype = (torch.bfloat16 if self.amp_enabled and torch.cuda.is_bf16_supported()
                          else torch.float16)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled and self.amp_dtype == torch.float16)
        self.group_size = config.batch_size * config.accumulation_steps
        self.total_steps = math.ceil(len(splits["train"]) / self.group_size) * config.epochs
        self.step, self.epoch, self.cursor, self.seen_tokens = 0, 0, 0, 0
        self.best_loss, self.last_eval_step = float("inf"), -1
        self.stop_requested = False

    def autocast(self):
        return torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled)

    def learning_rate(self):
        warmup = max(1, int(self.total_steps * self.config.warmup_ratio))
        if self.step < warmup:
            return self.config.learning_rate * (self.step + 1) / warmup
        progress = min(1.0, (self.step - warmup) / max(1, self.total_steps - warmup - 1))
        factor = self.config.min_lr_ratio + (1 - self.config.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
        return self.config.learning_rate * factor

    @torch.no_grad()
    def evaluate(self, split="validation", limit=None):
        was_training = self.model.training
        self.model.eval()
        samples = self.splits[split]
        samples = samples[:limit] if limit else samples
        loss_sum, token_count = 0.0, 0
        try:
            for start in range(0, len(samples), self.config.batch_size):
                x, y = collate_answers(samples[start:start + self.config.batch_size], self.tokenizer.pad_token_id)
                count = int((y != -100).sum())
                with self.autocast():
                    _, loss = self.model(x.to(self.device), y.to(self.device))
                loss_sum += loss.item() * count
                token_count += count
            result = loss_sum / token_count
            if not math.isfinite(result):
                raise ValueError("Non-finite evaluation loss")
            return result
        finally:
            self.model.train(was_training)

    def model_payload(self):
        return {"model_state_dict": self.model.state_dict(), "config": self.model.config,
                "step": self.step, "epoch": self.epoch, "val_loss": self.best_loss,
                "dialogue_format": getattr(self.model, "dialogue_format", "legacy"),
                "objective": getattr(self.model, "training_objective", "assistant_sft"),
                "data_fingerprint": self.data_fingerprint, "purpose": "Conversation training candidate; generated replies require review"}

    def save(self):
        payload = self.model_payload()
        payload.update({"training_config": asdict(self.config), "optimizer": self.optimizer.state_dict(),
                        "scaler": self.scaler.state_dict(), "cursor": self.cursor,
                        "seen_tokens": self.seen_tokens, "last_eval_step": self.last_eval_step,
                        "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None})
        atomic_save(payload, self.output_dir / "last.pt")

    def resume(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["training_config"] != asdict(self.config):
            raise ValueError("Resume settings differ from the original run; use the saved run settings")
        if payload["data_fingerprint"] != self.data_fingerprint:
            raise ValueError("Resume data differs from the original run")
        if payload["config"] != self.model.config:
            raise ValueError("Resume model architecture differs from the original run")
        if payload.get("dialogue_format", "legacy") != getattr(self.model, "dialogue_format", "legacy"):
            raise ValueError("Resume conversation format differs from the original run")
        if payload.get("objective", "assistant_sft") != getattr(self.model, "training_objective", "assistant_sft"):
            raise ValueError("Resume training objective differs from the original run")
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scaler.load_state_dict(payload["scaler"])
        self.step, self.epoch, self.cursor = payload["step"], payload["epoch"], payload["cursor"]
        self.seen_tokens, self.best_loss = payload["seen_tokens"], payload["val_loss"]
        self.last_eval_step = payload["last_eval_step"]
        torch.set_rng_state(payload["torch_rng"])
        random.setstate(payload["python_rng"])
        if payload["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])

    def event(self, record):
        with (self.output_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"step": self.step, **record}) + "\n")

    def validation_checkpoint(self):
        count = min(self.config.eval_examples, len(self.splits["validation"]))
        print(f"Validating {count:,} held-out answers at update {self.step}...", flush=True)
        loss = self.evaluate(limit=self.config.eval_examples)
        improved = loss < self.best_loss
        if improved:
            self.best_loss = loss
            atomic_save(self.model_payload(), self.output_dir / "best_chat.pt")
        print(f"VALIDATION step={self.step} loss={loss:.4f} best={self.best_loss:.4f}", flush=True)
        self.event({"kind": "validation", "loss": loss, "best": improved})
        if self.sample_callback:
            # Evaluation must not alter the dropout/sample RNG used by training.
            devices = [self.device.index or 0] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                review = self.sample_callback(self.model, self.step, self.output_dir)
            if review:
                self.event({"kind": "generation_review", **review})
                if review["status"] == "failed_generation_checks":
                    print(f"GENERATION CHECK FAILED: {review['failed_samples']}/{review['sample_count']} "
                          "sample replies are empty, repetitive, or reach the token limit without EOS. "
                          "A lower validation loss does not make this checkpoint usable.", flush=True)
                else:
                    print("Generation structure checks passed; replies still require manual correctness review.", flush=True)
        self.last_eval_step = self.step
        self.save()

    def fit(self, stop_after_steps=None):
        print(f"Device: {self.device} | precision: {str(self.amp_dtype) if self.amp_enabled else 'float32'}\n"
              f"Effective batch: {self.group_size} conversations/answers | planned updates: {self.total_steps:,}\n"
              "Ctrl+C requests a save after the current optimizer update.", flush=True)
        if self.step == 0 and self.last_eval_step < 0:
            self.validation_checkpoint()
        started, logged_tokens, logged_loss, logged_count = time.monotonic(), self.seen_tokens, 0.0, 0
        interval_start = started
        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch
            order = torch.randperm(len(self.splits["train"]),
                                   generator=torch.Generator().manual_seed(self.config.seed + epoch)).tolist()
            while self.cursor < len(order):
                if self.stop_requested:
                    self.save()
                    print(f"Paused safely at update {self.step}. Resume from {self.output_dir / 'last.pt'}", flush=True)
                    return
                end = min(self.cursor + self.group_size, len(order))
                samples = [self.splits["train"][i] for i in order[self.cursor:end]]
                # Weight each microbatch by supervised tokens, not padded length
                # or number of examples. This matches a single combined batch.
                total_tokens = sum(len(s["tokens"]) - 1 - s["answer_start"] for s in samples)
                rate = self.learning_rate()
                for group in self.optimizer.param_groups:
                    group["lr"] = rate
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                group_loss = 0.0
                for start in range(0, len(samples), self.config.batch_size):
                    x, y = collate_answers(samples[start:start + self.config.batch_size], self.tokenizer.pad_token_id)
                    count = int((y != -100).sum())
                    with self.autocast():
                        _, loss = self.model(x.to(self.device), y.to(self.device))
                        weighted = loss * (count / total_tokens)
                    if not torch.isfinite(loss):
                        raise ValueError(f"Non-finite training loss at update {self.step}; resume from last.pt")
                    self.scaler.scale(weighted).backward()
                    group_loss += loss.item() * count
                self.scaler.unscale_(self.optimizer)
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise ValueError(f"Non-finite gradients at update {self.step}; resume from last.pt")
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.step += 1
                self.cursor = end
                self.seen_tokens += total_tokens
                logged_loss += group_loss
                logged_count += total_tokens
                if self.step % self.config.log_every == 0 or end == len(order):
                    elapsed = time.monotonic() - interval_start
                    speed = (self.seen_tokens - logged_tokens) / max(elapsed, 1e-6)
                    print(f"epoch {epoch + 1}/{self.config.epochs} | update {self.step:,}/{self.total_steps:,} | "
                          f"loss {logged_loss / logged_count:.4f} | lr {rate:.2e} | "
                          f"supervised tokens/s {speed:,.0f}" +
                          (f" | peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB" if self.device.type == "cuda" else ""), flush=True)
                    self.event({"kind": "train", "loss": logged_loss / logged_count, "lr": rate,
                                "answer_tokens_per_second": speed, "seen_answer_tokens": self.seen_tokens})
                    interval_start, logged_tokens = time.monotonic(), self.seen_tokens
                    logged_loss, logged_count = 0.0, 0
                if self.step % self.config.eval_every == 0:
                    self.validation_checkpoint()
                    interval_start = time.monotonic()  # Exclude validation from throughput.
                elif self.step % self.config.save_every == 0:
                    self.save()
                if stop_after_steps and self.step >= stop_after_steps:
                    self.save()
                    return
            self.epoch, self.cursor = epoch + 1, 0
            if self.last_eval_step != self.step:
                self.validation_checkpoint()
            else:
                self.save()
        print(f"Optimization complete. Lowest-validation-loss candidate: {self.output_dir / 'best_chat.pt'}\n"
              "This does not certify chatbot quality. Review the generated replies before using the checkpoint.", flush=True)
