"""Framework-agnostic execution backend interface.

v1 ships only `PyTorchBackend`. JAX/TensorFlow backends would implement this same
protocol; none exist yet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..strategies import ExecutionStrategy


class ExecutionBackend(ABC):
    @abstractmethod
    def training_step(self, batch: Any, loss_fn: Any) -> float:
        """Run one training step (forward + backward + optimizer update as appropriate
        given accumulation state) and return the scalar loss value."""

    @abstractmethod
    def apply_strategy(self, strategy: ExecutionStrategy) -> None:
        """Reconfigure execution (precision, batch size, grad accumulation,
        checkpointing) to match the given strategy."""

    @abstractmethod
    def evaluate(self, val_loader: Any) -> float | None:
        """Run validation and return an accuracy-like metric (higher is better),
        or None if it cannot be computed."""
