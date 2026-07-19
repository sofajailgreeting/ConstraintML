"""ConstraintPlanner: a deterministic, explainable, rule-based policy layer.

This is a first-pass heuristic, not a learned/ML optimizer. Every evaluation
projects cumulative resource usage to the end of the run, compares it against
the user's budgets, and -- if any budget is trending over -- applies the first
untried lever in a fixed severity ladder. Behavior is intentionally monotonic
(no stepping back down once a lever has been used) to avoid oscillation; that
recovery path is documented future work, not a v1 feature.
"""

from __future__ import annotations

from dataclasses import replace

from ..constraints.spec import ConstraintSpec
from .state import RunState
from .strategies import ExecutionStrategy, PlannerDecision, StrategyAction

_MAX_GRAD_ACCUMULATION_STEPS = 8
_MAX_BATCH_SIZE_MULTIPLIER = 4
_BATCH_SIZE_GROWTH_FACTOR = 1.5
_EARLY_STOP_HYSTERESIS_EVALUATIONS = 2


class ConstraintPlanner:
    def __init__(
        self,
        spec: ConstraintSpec,
        *,
        evaluate_every_n_steps: int = 50,
        warning_threshold: float = 0.85,
        critical_threshold: float = 1.0,
        accuracy_guard_buffer: float = 0.8,
        expected_total_steps: int | None = None,
    ):
        self.spec = spec
        self.evaluate_every_n_steps = evaluate_every_n_steps
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.accuracy_guard_buffer = accuracy_guard_buffer
        self.expected_total_steps = expected_total_steps

        self._accuracy_guard_tripped = False
        self._consecutive_overages = 0
        self._initial_batch_size: int | None = None

    def should_evaluate(self, state: RunState) -> bool:
        return state.step_count > 0 and state.step_count % self.evaluate_every_n_steps == 0

    def _project(self, cumulative: float, state: RunState) -> float:
        # Extrapolates the CURRENT per-step rate out to expected_total_steps. Since the
        # per-step rate is roughly constant across a run, this projection is roughly
        # constant from the very first evaluation onward - it does not gradually climb
        # toward the budget over time. A tight energy/carbon/cost budget therefore tends
        # to trigger the ladder almost immediately (at whatever rate training happens to
        # run), not partway through the run. `deadline`, by contrast, compares live
        # elapsed_seconds directly (see _fractions below) and so genuinely ramps from 0
        # to 1 as the run progresses: it is the right lever for "stop once most of the
        # allotted budget window has passed."
        if self.expected_total_steps and state.step_count > 0:
            return cumulative * (self.expected_total_steps / state.step_count)
        return cumulative

    def _fractions(self, state: RunState) -> dict[str, float]:
        fractions: dict[str, float] = {}
        spec = self.spec
        if spec.energy_budget_kwh:
            fractions["energy"] = self._project(state.cumulative_energy_kwh, state) / spec.energy_budget_kwh
        if spec.carbon_budget_kgco2e:
            fractions["carbon"] = self._project(state.cumulative_carbon_kgco2e, state) / spec.carbon_budget_kgco2e
        if spec.cost_budget_usd:
            fractions["cost"] = self._project(state.cumulative_cost_usd, state) / spec.cost_budget_usd
        if spec.deadline_seconds:
            # Time is already a live measurement, not a per-step accumulator, so it is
            # never extrapolated the way resource totals are.
            fractions["time"] = state.elapsed_seconds / spec.deadline_seconds
        return fractions

    def _update_accuracy_guard(self, state: RunState) -> None:
        max_loss = self.spec.max_accuracy_loss
        loss = state.accuracy_loss()
        if max_loss is not None and loss is not None and loss >= max_loss * self.accuracy_guard_buffer:
            self._accuracy_guard_tripped = True

    def decide(self, state: RunState) -> PlannerDecision:
        fractions = self._fractions(state)
        current = state.current_strategy
        if self._initial_batch_size is None:
            self._initial_batch_size = current.batch_size or 1

        if not fractions:
            return PlannerDecision(StrategyAction.NO_CHANGE, "No budgets configured.", current)

        self._update_accuracy_guard(state)

        worst_metric = max(fractions, key=lambda metric: (fractions[metric], metric == self.spec.optimize_for))
        worst_fraction = fractions[worst_metric]

        if worst_fraction < self.warning_threshold:
            self._consecutive_overages = 0
            return PlannerDecision(
                StrategyAction.NO_CHANGE,
                f"{worst_metric} projected at {worst_fraction:.0%} of budget, under warning threshold.",
                current,
            )

        self._consecutive_overages += 1
        critical = worst_fraction >= self.critical_threshold
        reason_prefix = f"{worst_metric} projected at {worst_fraction:.0%} of budget"

        lever = self._next_lever(state, current)
        if lever is not None:
            action, new_strategy = lever
            if action is StrategyAction.REDUCE_PRECISION:
                state.precision_reduced_at_step = state.step_count
            return PlannerDecision(action, f"{reason_prefix}; applying next lever ({action.value}).", new_strategy)

        allow_early_stop = critical or self._consecutive_overages >= _EARLY_STOP_HYSTERESIS_EVALUATIONS
        if allow_early_stop:
            reason = f"{reason_prefix}; all optimization levers exhausted."
            state.stop_reason = reason
            return PlannerDecision(StrategyAction.EARLY_STOP, reason, current)

        return PlannerDecision(
            StrategyAction.NO_CHANGE,
            f"{reason_prefix}; levers exhausted, awaiting repeated overage before stopping.",
            current,
        )

    def _next_lever(
        self, state: RunState, current: ExecutionStrategy
    ) -> tuple[StrategyAction, ExecutionStrategy] | None:
        if current.precision != "fp16" and not self._accuracy_guard_tripped:
            return StrategyAction.REDUCE_PRECISION, replace(current, precision="fp16")

        if not current.activation_checkpointing:
            return StrategyAction.ENABLE_CHECKPOINTING, replace(current, activation_checkpointing=True)

        max_batch_size = self._initial_batch_size * _MAX_BATCH_SIZE_MULTIPLIER
        if current.batch_size < max_batch_size:
            grown = max(current.batch_size + 1, int(current.batch_size * _BATCH_SIZE_GROWTH_FACTOR))
            new_batch_size = min(grown, max_batch_size)
            return StrategyAction.INCREASE_BATCH_SIZE, replace(current, batch_size=new_batch_size)

        if current.grad_accumulation_steps < _MAX_GRAD_ACCUMULATION_STEPS:
            return (
                StrategyAction.INCREASE_GRAD_ACCUMULATION,
                replace(current, grad_accumulation_steps=current.grad_accumulation_steps + 1),
            )

        return None
