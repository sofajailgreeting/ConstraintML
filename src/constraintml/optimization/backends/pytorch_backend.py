"""PyTorch execution backend: the only real ExecutionBackend implementation for v1."""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader, RandomSampler

from ..strategies import ExecutionStrategy
from .base import ExecutionBackend

logger = logging.getLogger(__name__)


class PyTorchBackend(ExecutionBackend):
    def __init__(self, model, optimizer, train_loader, loss_fn, device=None):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        initial_batch_size = getattr(train_loader, "batch_size", None) or 0
        self.strategy = ExecutionStrategy(batch_size=initial_batch_size)
        self._initial_batch_size = initial_batch_size
        self._micro_step = 0
        self._checkpoint_warned = False

        self.optimizer.zero_grad()

    def apply_strategy(self, strategy: ExecutionStrategy) -> None:
        if strategy.batch_size and strategy.batch_size != self.strategy.batch_size:
            self._resize_batch(strategy.batch_size)
        self.strategy = strategy

    def _resize_batch(self, batch_size: int) -> None:
        dataset = getattr(self.train_loader, "dataset", None)
        if dataset is None:
            logger.warning(
                "Cannot resize batch size to %d: train_loader has no `.dataset` "
                "(only map-style DataLoaders support runtime batch-size changes in v1).",
                batch_size,
            )
            return
        shuffled = isinstance(getattr(self.train_loader, "sampler", None), RandomSampler)
        self.train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffled,
            num_workers=getattr(self.train_loader, "num_workers", 0),
        )

    def training_step(self, batch, loss_fn=None) -> float:
        loss_fn = loss_fn or self.loss_fn
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        autocast_enabled = self.strategy.precision == "fp16"
        # CPU has no native fp16 compute kernels -- torch.autocast(dtype=float16) on CPU
        # falls back to emulation and measured ~44x SLOWER than fp32 for a mid-sized MLP
        # on this machine, actively defeating the time/energy budget instead of helping it.
        # bfloat16 is the dtype CPU autocast is actually designed around; fp16 remains
        # correct for CUDA, where it's backed by real tensor-core throughput.
        autocast_dtype = torch.bfloat16 if self.device.type == "cpu" else torch.float16
        with torch.autocast(device_type=self.device.type, enabled=autocast_enabled, dtype=autocast_dtype):
            outputs = self._forward(inputs)
            loss = loss_fn(outputs, targets)

        (loss / self.strategy.grad_accumulation_steps).backward()
        self._micro_step += 1

        if self._micro_step % self.strategy.grad_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item()

    def _forward(self, inputs):
        if self.strategy.activation_checkpointing:
            if isinstance(self.model, torch.nn.Sequential) and len(self.model) > 1:
                return torch.utils.checkpoint.checkpoint_sequential(
                    self.model, len(self.model), inputs, use_reentrant=False
                )
            if not self._checkpoint_warned:
                logger.warning(
                    "Activation checkpointing requested but the model is not an "
                    "nn.Sequential, so its forward pass cannot be automatically "
                    "wrapped in v1. Continuing without checkpointing."
                )
                self._checkpoint_warned = True
        return self.model(inputs)

    def evaluate(self, val_loader) -> float | None:
        if val_loader is None or self.loss_fn is None:
            return None

        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                total_loss += self.loss_fn(outputs, targets).item()
                total_batches += 1
        self.model.train()

        if total_batches == 0:
            return None
        # "Accuracy-like" metric where higher is better; without a task-specific
        # metric available, negative average validation loss becomes the generic proxy.
        return -(total_loss / total_batches)
