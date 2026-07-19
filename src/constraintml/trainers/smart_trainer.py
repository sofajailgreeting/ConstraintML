"""SmartTrainer: pick a primary optimization objective, budgets act as guardrails."""

from __future__ import annotations

from numbers import Number

from ..constraints.spec import OPTIMIZE_FOR_CHOICES
from .base import ConstraintTrainer


class SmartTrainer(ConstraintTrainer):
    """`optimize_for` simply selects the metric used as a tie-breaker when multiple budgets
    are simultaneously trending over - it does not change the planner's ladder logic,
    only which metric is treated as most urgent when severities are close."""

    def __init__(
        self,
        model=None,
        optimizer=None,
        *,
        optimize_for: str = "carbon",
        carbon_budget: str | Number | None = None,
        energy_budget: str | Number | None = None,
        deadline: str | Number | None = None,
        max_accuracy_loss: str | Number | None = None,
        cost_budget: str | Number | None = None,
        **kwargs,
    ):
        if optimize_for not in OPTIMIZE_FOR_CHOICES:
            raise ValueError(f"optimize_for must be one of {OPTIMIZE_FOR_CHOICES}, got {optimize_for!r}")
        super().__init__(
            model=model,
            optimizer=optimizer,
            energy_budget=energy_budget,
            carbon_budget=carbon_budget,
            deadline=deadline,
            max_accuracy_loss=max_accuracy_loss,
            cost_budget=cost_budget,
            optimize_for=optimize_for,
            **kwargs,
        )
