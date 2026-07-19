"""GreenTrainer: maximize accuracy subject to fixed resource budgets."""

from __future__ import annotations

from numbers import Number

from .base import ConstraintTrainer


class GreenTrainer(ConstraintTrainer):
    """Always optimizes for accuracy subject to the given energy/carbon/deadline
    budgets -- unlike SmartTrainer, it does not expose `optimize_for`."""

    def __init__(
        self,
        model=None,
        optimizer=None,
        *,
        energy_budget: str | Number | None = None,
        carbon_budget: str | Number | None = None,
        deadline: str | Number | None = None,
        max_accuracy_loss: str | Number | None = None,
        cost_budget: str | Number | None = None,
        **kwargs,
    ):
        kwargs.pop("optimize_for", None)
        super().__init__(
            model=model,
            optimizer=optimizer,
            energy_budget=energy_budget,
            carbon_budget=carbon_budget,
            deadline=deadline,
            max_accuracy_loss=max_accuracy_loss,
            cost_budget=cost_budget,
            optimize_for="accuracy",
            **kwargs,
        )
