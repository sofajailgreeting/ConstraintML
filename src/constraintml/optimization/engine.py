"""RuntimeOptimizationEngine: orchestrates telemetry, carbon accounting, and the
planner around each training step, applying any resulting strategy to the backend.
"""

from __future__ import annotations

import time

from .backends.base import ExecutionBackend
from .carbon import CarbonModel
from .planner import ConstraintPlanner
from .state import RunState
from .strategies import StrategyAction
from .telemetry.base import TelemetryCollector, TelemetrySample


class RuntimeOptimizationEngine:
    def __init__(
        self,
        planner: ConstraintPlanner,
        telemetry: TelemetryCollector,
        backend: ExecutionBackend,
        carbon_model: CarbonModel | None = None,
        cost_per_kwh_usd: float | None = None,
        state: RunState | None = None,
    ):
        self.planner = planner
        self.telemetry = telemetry
        self.backend = backend
        self.carbon_model = carbon_model or CarbonModel()
        self.cost_per_kwh_usd = cost_per_kwh_usd
        self.state = state or RunState()

        self._step_start_time: float | None = None
        self._step_start_sample: TelemetrySample | None = None

    def begin_step(self) -> None:
        self._step_start_time = time.perf_counter()
        self._step_start_sample = self.telemetry.sample()

    def end_step(self, loss: float | None = None) -> None:
        if self._step_start_time is None or self._step_start_sample is None:
            raise RuntimeError("end_step() called without a matching begin_step().")

        end_sample = self.telemetry.sample()
        elapsed = time.perf_counter() - self._step_start_time

        energy_kwh = self.telemetry.integrate_energy(self._step_start_sample, end_sample, elapsed)
        carbon_kgco2e = self.carbon_model.to_carbon(energy_kwh)

        self.state.cumulative_energy_kwh += energy_kwh
        self.state.cumulative_carbon_kgco2e += carbon_kgco2e
        if self.cost_per_kwh_usd is not None:
            self.state.cumulative_cost_usd += energy_kwh * self.cost_per_kwh_usd
        self.state.elapsed_seconds += elapsed
        self.state.step_count += 1

        self._step_start_time = None
        self._step_start_sample = None

        if self.planner.should_evaluate(self.state):
            decision = self.planner.decide(self.state)
            self._apply_decision(decision)

    def _apply_decision(self, decision) -> None:
        if decision.action is StrategyAction.EARLY_STOP:
            self.state.stopped_early = True
            self.state.stop_reason = decision.reason
            return
        if decision.action is StrategyAction.NO_CHANGE:
            return
        self.backend.apply_strategy(decision.resulting_strategy)
        self.state.record_strategy(decision.resulting_strategy)

    def should_stop(self) -> bool:
        return self.state.stopped_early
