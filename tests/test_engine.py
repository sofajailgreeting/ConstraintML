import time

from constraintml.optimization.backends.base import ExecutionBackend
from constraintml.optimization.carbon import CarbonModel
from constraintml.optimization.engine import RuntimeOptimizationEngine
from constraintml.optimization.planner import ConstraintPlanner
from constraintml.constraints.spec import ConstraintSpec
from constraintml.optimization.strategies import ExecutionStrategy, PlannerDecision, StrategyAction
from constraintml.optimization.telemetry.base import TelemetryCollector, TelemetrySample


class FakeTelemetryCollector(TelemetryCollector):
    """Deterministic telemetry: constant power, no real clock dependency needed
    since engine.end_step() uses wall-clock elapsed time internally."""

    def __init__(self, power_watts=360.0):
        self.power_watts = power_watts

    def sample(self):
        return TelemetrySample(timestamp=0.0, power_watts=self.power_watts)


class FakeBackend(ExecutionBackend):
    def __init__(self):
        self.applied_strategies = []

    def training_step(self, batch, loss_fn):
        return 0.0

    def apply_strategy(self, strategy):
        self.applied_strategies.append(strategy)

    def evaluate(self, val_loader):
        return None


class FixedPlanner:
    """Test double standing in for ConstraintPlanner with a scripted decision sequence."""

    def __init__(self, decisions, evaluate_every_n_steps=1):
        self._decisions = iter(decisions)
        self.evaluate_every_n_steps = evaluate_every_n_steps
        self.evaluations = 0

    def should_evaluate(self, state):
        return state.step_count > 0 and state.step_count % self.evaluate_every_n_steps == 0

    def decide(self, state):
        self.evaluations += 1
        return next(self._decisions)


def test_end_step_accumulates_energy_and_carbon():
    telemetry = FakeTelemetryCollector(power_watts=3600.0)  # 3600W constant
    backend = FakeBackend()
    planner = ConstraintPlanner(ConstraintSpec(), evaluate_every_n_steps=1000)
    engine = RuntimeOptimizationEngine(
        planner=planner, telemetry=telemetry, backend=backend, carbon_model=CarbonModel(intensity_kg_per_kwh=0.5)
    )

    engine.begin_step()
    time.sleep(0.01)  # ensure a measurable, non-zero elapsed time for the energy integral
    engine.end_step(loss=1.0)

    assert engine.state.step_count == 1
    assert engine.state.cumulative_energy_kwh > 0
    assert engine.state.cumulative_carbon_kgco2e == engine.state.cumulative_energy_kwh * 0.5


def test_engine_applies_planner_decision_and_records_strategy():
    telemetry = FakeTelemetryCollector()
    backend = FakeBackend()
    new_strategy = ExecutionStrategy(precision="fp16")
    decision = PlannerDecision(StrategyAction.REDUCE_PRECISION, "test", new_strategy)
    planner = FixedPlanner([decision], evaluate_every_n_steps=1)

    engine = RuntimeOptimizationEngine(planner=planner, telemetry=telemetry, backend=backend)
    engine.begin_step()
    engine.end_step()

    assert backend.applied_strategies == [new_strategy]
    assert engine.state.current_strategy == new_strategy
    assert engine.state.strategy_history == [(1, new_strategy)]


def test_engine_stops_on_early_stop_decision():
    telemetry = FakeTelemetryCollector()
    backend = FakeBackend()
    decision = PlannerDecision(StrategyAction.EARLY_STOP, "budget exceeded", ExecutionStrategy())
    planner = FixedPlanner([decision], evaluate_every_n_steps=1)

    engine = RuntimeOptimizationEngine(planner=planner, telemetry=telemetry, backend=backend)
    engine.begin_step()
    engine.end_step()

    assert engine.should_stop() is True
    assert engine.state.stop_reason == "budget exceeded"
    assert backend.applied_strategies == []


def test_engine_skips_evaluation_before_cadence():
    telemetry = FakeTelemetryCollector()
    backend = FakeBackend()
    planner = FixedPlanner([], evaluate_every_n_steps=5)

    engine = RuntimeOptimizationEngine(planner=planner, telemetry=telemetry, backend=backend)
    for _ in range(4):
        engine.begin_step()
        engine.end_step()

    assert planner.evaluations == 0
    assert engine.should_stop() is False


def test_cost_tracking_only_when_rate_configured():
    telemetry = FakeTelemetryCollector(power_watts=3600.0)
    backend = FakeBackend()
    planner = ConstraintPlanner(ConstraintSpec(), evaluate_every_n_steps=1000)
    engine = RuntimeOptimizationEngine(
        planner=planner, telemetry=telemetry, backend=backend, cost_per_kwh_usd=0.20
    )

    engine.begin_step()
    engine.end_step()

    assert engine.state.cumulative_cost_usd == engine.state.cumulative_energy_kwh * 0.20
