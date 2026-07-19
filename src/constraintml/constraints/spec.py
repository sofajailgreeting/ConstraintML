"""ConstraintSpec: the normalized, internal representation of user-supplied budgets."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number

from .parsing import parse_carbon, parse_cost, parse_duration, parse_energy, parse_percentage

OPTIMIZE_FOR_CHOICES = ("accuracy", "energy", "carbon", "cost", "time")

_DICT_KEY_ALIASES = {
    "energy": "energy_budget",
    "energy_budget": "energy_budget",
    "carbon": "carbon_budget",
    "carbon_budget": "carbon_budget",
    "deadline": "deadline",
    "max_accuracy_loss": "max_accuracy_loss",
    "accuracy_loss": "max_accuracy_loss",
    "cost": "cost_budget",
    "cost_budget": "cost_budget",
}


@dataclass(frozen=True)
class ConstraintSpec:
    energy_budget_kwh: float | None = None
    carbon_budget_kgco2e: float | None = None
    deadline_seconds: float | None = None
    max_accuracy_loss: float | None = None
    cost_budget_usd: float | None = None
    optimize_for: str = "accuracy"

    def __post_init__(self):
        if self.optimize_for not in OPTIMIZE_FOR_CHOICES:
            raise ValueError(
                f"optimize_for must be one of {OPTIMIZE_FOR_CHOICES}, got {self.optimize_for!r}"
            )

    @classmethod
    def from_kwargs(
        cls,
        *,
        energy_budget: str | Number | None = None,
        carbon_budget: str | Number | None = None,
        deadline: str | Number | None = None,
        max_accuracy_loss: str | Number | None = None,
        cost_budget: str | Number | None = None,
        optimize_for: str = "accuracy",
    ) -> "ConstraintSpec":
        return cls(
            energy_budget_kwh=parse_energy(energy_budget) if energy_budget is not None else None,
            carbon_budget_kgco2e=parse_carbon(carbon_budget) if carbon_budget is not None else None,
            deadline_seconds=parse_duration(deadline) if deadline is not None else None,
            max_accuracy_loss=parse_percentage(max_accuracy_loss) if max_accuracy_loss is not None else None,
            cost_budget_usd=parse_cost(cost_budget) if cost_budget is not None else None,
            optimize_for=optimize_for,
        )

    @classmethod
    def from_dict(cls, constraints: dict, optimize_for: str = "accuracy") -> "ConstraintSpec":
        kwargs: dict = {}
        for key, value in constraints.items():
            try:
                field = _DICT_KEY_ALIASES[key]
            except KeyError:
                raise ValueError(f"Unknown constraint key: {key!r}") from None
            kwargs[field] = value
        return cls.from_kwargs(optimize_for=optimize_for, **kwargs)

    def merge(self, other: "ConstraintSpec") -> "ConstraintSpec":
        """Return a spec with `other`'s non-None fields overriding this spec's fields.

        Used to let named kwargs (e.g. GreenTrainer(energy_budget=...)) win over a
        `constraints={...}` dict passed to the same constructor.
        """
        merged = {}
        for field_name in self.__dataclass_fields__:
            other_value = getattr(other, field_name)
            merged[field_name] = other_value if other_value is not None else getattr(self, field_name)
        return ConstraintSpec(**merged)
