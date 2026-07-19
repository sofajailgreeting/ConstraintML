"""RunState: the mutable, cumulative record of a single training run."""

from __future__ import annotations

from dataclasses import dataclass, field

from .strategies import ExecutionStrategy


@dataclass(frozen=True)
class EpochSnapshot:
    epoch: int
    cumulative_energy_kwh: float
    cumulative_carbon_kgco2e: float
    elapsed_seconds: float
    val_accuracy: float | None


@dataclass
class RunState:
    cumulative_energy_kwh: float = 0.0
    cumulative_carbon_kgco2e: float = 0.0
    cumulative_cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    step_count: int = 0
    epoch: int = 0

    baseline_val_accuracy: float | None = None
    current_val_accuracy: float | None = None

    current_strategy: ExecutionStrategy = field(default_factory=ExecutionStrategy)
    strategy_history: list[tuple[int, ExecutionStrategy]] = field(default_factory=list)
    epoch_history: list[EpochSnapshot] = field(default_factory=list)

    stopped_early: bool = False
    stop_reason: str | None = None

    # Set by the planner the first time it reduces precision; used by the accuracy guard
    # to know a precision change is the (most likely) cause of an accuracy regression.
    precision_reduced_at_step: int | None = None

    def accuracy_loss(self) -> float | None:
        if self.baseline_val_accuracy is None or self.current_val_accuracy is None:
            return None
        return max(0.0, self.baseline_val_accuracy - self.current_val_accuracy)

    def record_strategy(self, strategy: ExecutionStrategy) -> None:
        self.current_strategy = strategy
        self.strategy_history.append((self.step_count, strategy))
