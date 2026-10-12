"""Repeatable evaluation harness for Robbie models, token generation, and context retention (Priority 4)."""

from dataclasses import dataclass, field
import math
import time
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn


@dataclass
class ModelEvaluationResult:
    """Summary of model evaluation metrics and benchmark thresholds."""
    val_loss: Optional[float] = None
    perplexity: Optional[float] = None
    distinct_1: float = 0.0
    distinct_2: float = 0.0
    tokens_per_second: float = 0.0
    context_recall_rate: float = 1.0
    scenarios_tested: int = 0
    passed: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "val_loss": round(self.val_loss, 4) if self.val_loss else None,
            "perplexity": round(self.perplexity, 2) if self.perplexity else None,
            "distinct_1": round(self.distinct_1, 3),
            "distinct_2": round(self.distinct_2, 3),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "context_recall_rate": round(self.context_recall_rate, 3),
            "scenarios_tested": self.scenarios_tested,
            "passed": self.passed,
            "details": self.details,
        }


class ModelEvaluator:
    """Evaluates language model quality, diversity, latency, and conversation retention."""

    @staticmethod
    def distinct_ngrams(text: str, n: int = 1) -> float:
        """Compute ratio of unique n-grams to total n-grams."""
        tokens = text.split()
        if len(tokens) < n:
            return 1.0
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 1.0
        return len(set(ngrams)) / len(ngrams)

    @classmethod
    def evaluate_diversity(cls, sample_texts: List[str]) -> Tuple[float, float]:
        """Compute average Distinct-1 and Distinct-2 scores across generations."""
        if not sample_texts:
            return 0.0, 0.0
        d1 = [cls.distinct_ngrams(t, 1) for t in sample_texts]
        d2 = [cls.distinct_ngrams(t, 2) for t in sample_texts]
        return sum(d1) / len(d1), sum(d2) / len(d2)

    @classmethod
    def evaluate_perplexity(
        cls,
        model: nn.Module,
        dataloader: Any,
        device: torch.device,
        max_batches: int = 50,
    ) -> Tuple[float, float]:
        """Compute cross-entropy loss and perplexity on evaluation batches."""
        model.eval()
        total_loss = 0.0
        total_batches = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                elif isinstance(batch, dict):
                    inputs, targets = batch["input_ids"].to(device), batch["labels"].to(device)
                else:
                    continue

                logits = model(inputs)
                if isinstance(logits, tuple):
                    logits = logits[0]

                # Reshape for cross entropy: (batch * seq, vocab) vs (batch * seq)
                vocab_size = logits.size(-1)
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                total_loss += loss.item()
                total_batches += 1

        if total_batches == 0:
            return 0.0, 1.0

        avg_loss = total_loss / total_batches
        perplexity = math.exp(min(avg_loss, 20.0))
        return avg_loss, perplexity

    @classmethod
    def evaluate_multi_turn_retention(cls, engine: Any) -> Dict[str, Any]:
        """Evaluate slot tracking and persona continuity over a multi-turn conversation."""
        engine.context.clear(keep_slots=False, keep_persistent=False)

        scenario = [
            ("Hello Robbie!", "greeting"),
            ("My name is Alex", "slot_intro"),
            ("What is my name?", "slot_recall"),
            ("I work as a roboticist", "role_intro"),
            ("What is my role?", "role_recall"),
        ]

        turns_log = []
        name_recalled = False
        role_recalled = False

        for user_msg, intent in scenario:
            t0 = time.perf_counter()
            reply = engine.respond(user_msg)
            elapsed = time.perf_counter() - t0

            if intent == "slot_recall" and "Alex" in reply:
                name_recalled = True
            if intent == "role_recall" and "roboticist" in reply.lower():
                role_recalled = True

            turns_log.append({
                "intent": intent,
                "user": user_msg,
                "reply": reply,
                "latency_sec": elapsed,
            })

        recall_count = (1 if name_recalled else 0) + (1 if role_recalled else 0)
        recall_rate = recall_count / 2.0

        return {
            "name_recalled": name_recalled,
            "role_recalled": role_recalled,
            "recall_rate": recall_rate,
            "turns_log": turns_log,
        }

    @classmethod
    def run_benchmark(
        cls,
        engine: Any,
        sample_prompts: Optional[List[str]] = None,
    ) -> ModelEvaluationResult:
        """Run repeatable model benchmark evaluating diversity, latency, and retention."""
        prompts = sample_prompts or [
            "Explain self-attention in two sentences.",
            "What is validation perplexity?",
            "How does BPE tokenization work?",
        ]

        replies: List[str] = []
        total_tokens = 0
        total_time = 0.0

        for p in prompts:
            t0 = time.perf_counter()
            rep = engine.respond(p)
            dt = time.perf_counter() - t0
            replies.append(rep)
            total_time += dt
            if hasattr(engine, "tokenizer") and hasattr(engine.tokenizer, "encode"):
                total_tokens += len(engine.tokenizer.encode(rep, add_special_tokens=False))
            else:
                total_tokens += len(rep.split())

        avg_d1, avg_d2 = cls.evaluate_diversity(replies)
        toks_per_sec = (total_tokens / total_time) if total_time > 0 else 0.0

        retention = cls.evaluate_multi_turn_retention(engine)

        passed = (avg_d1 >= 0.40) and (retention["recall_rate"] >= 0.50)

        return ModelEvaluationResult(
            distinct_1=avg_d1,
            distinct_2=avg_d2,
            tokens_per_second=toks_per_sec,
            context_recall_rate=retention["recall_rate"],
            scenarios_tested=len(prompts) + 1,
            passed=passed,
            details={
                "retention": retention,
                "prompts_count": len(prompts),
                "total_generated_tokens": total_tokens,
            },
        )
