"""Training engine, validation loops, and checkpointing for Project Robbie."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import torch
from torch.utils.data import DataLoader

from robbie.utils.logger import logger
from robbie.utils.device import get_best_device


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Linear warmup followed by cosine annealing decay to min_lr_ratio."""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@dataclass
class TrainingConfig:
    """Hyperparameters for training Robbie models."""
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 5e-4
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    use_amp: bool = True               # Automatic Mixed Precision (FP16 on CUDA)
    use_cosine_lr: bool = True         # Learning rate warmup + cosine decay
    checkpoint_dir: str = "checkpoints"
    best_model_name: str = "best_model.pt"
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 1e-4
    device: str = "auto"


class Trainer:
    """Orchestrates model training, evaluation, and checkpoint management."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset,
        val_dataset,
        config: Optional[TrainingConfig] = None,
    ):
        self.config = config or TrainingConfig()
        self.device = get_best_device() if self.config.device == "auto" else self.config.device
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Automatic Mixed Precision (AMP) on CUDA
        is_cuda = self.device == "cuda" or (isinstance(self.device, torch.device) and self.device.type == "cuda")
        self.use_amp = self.config.use_amp and is_cuda
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, dataloader: DataLoader, scheduler: Optional[Any] = None) -> float:
        """Run one training epoch with optional AMP and LR scheduling."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                _, loss = self.model(x, targets=y)

            if loss is not None:
                self.scaler.scale(loss).backward()

                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                if scheduler is not None:
                    scheduler.step()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate validation loss with optional AMP."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                _, loss = self.model(x, targets=y)
            if loss is not None:
                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(1, num_batches)

    def fit(self) -> Dict[str, Any]:
        """Execute full training loop with scheduler and early stopping."""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True if len(self.train_dataset) > self.config.batch_size else False,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        scheduler = None
        if self.config.use_cosine_lr and total_steps > 0:
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=self.config.min_lr_ratio,
            )

        history: Dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "val_perplexity": [],
            "learning_rates": [],
            "best_epoch": 1,
            "best_val_loss": float("inf"),
            "best_val_perplexity": float("inf"),
        }
        best_val_loss = float("inf")
        best_epoch = 1
        patience_counter = 0

        logger.info(f"Starting training on device: {str(self.device).upper()}")
        logger.info(
            f"Epochs: {self.config.epochs} | Batch Size: {self.config.batch_size} | "
            f"Base LR: {self.config.learning_rate} | AMP: {self.use_amp} | Cosine LR: {self.config.use_cosine_lr}"
        )
        if self.config.early_stopping_patience is not None:
            logger.info(f"Early Stopping enabled with patience: {self.config.early_stopping_patience} epochs")

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader, scheduler=scheduler)
            val_loss = self.evaluate(val_loader)
            val_ppl = math.exp(min(val_loss, 20.0))
            current_lr = self.optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_perplexity"].append(val_ppl)
            history["learning_rates"].append(current_lr)

            logger.info(
                f"Epoch [{epoch}/{self.config.epochs}] | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.2f} | LR: {current_lr:.6f}"
            )

            # Check if this is a new best checkpoint
            if val_loss < (best_val_loss - self.config.early_stopping_min_delta):
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(
                    self.checkpoint_dir / self.config.best_model_name,
                    epoch=epoch,
                    val_loss=val_loss,
                )
            else:
                patience_counter += 1
                if (
                    self.config.early_stopping_patience is not None
                    and patience_counter >= self.config.early_stopping_patience
                ):
                    logger.info(
                        f"Early stopping triggered at epoch {epoch}! Best val loss was {best_val_loss:.4f} (PPL: {math.exp(min(best_val_loss, 20.0)):.2f}) at epoch {best_epoch}."
                    )
                    break

        history["best_epoch"] = best_epoch
        history["best_val_loss"] = best_val_loss
        history["best_val_perplexity"] = math.exp(min(best_val_loss, 20.0))

        # Save final checkpoint
        final_path = self.checkpoint_dir / "latest_model.pt"
        self.save_checkpoint(final_path, epoch=epoch, val_loss=val_loss)

        # Save training loss history to JSON
        import json
        history_file = self.checkpoint_dir / "training_history.json"
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        return history

    def save_checkpoint(
        self,
        filepath: Path,
        epoch: int,
        val_loss: float,
    ) -> None:
        """Save model weights, optimizer state, and metadata."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "val_loss": val_loss,
            "val_perplexity": math.exp(min(val_loss, 20.0)),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": getattr(self.model, "config", None),
        }
        torch.save(checkpoint, filepath)

    @staticmethod
    def load_checkpoint(
        filepath: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint weights into model and optional optimizer."""
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint
