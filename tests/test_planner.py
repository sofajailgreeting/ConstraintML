import pytest

from constraintml.constraints.spec import ConstraintSpec
from constraintml.optimization.planner import ConstraintPlanner
from constraintml.optimization.state import RunState
from constraintml.optimization.strategies import ExecutionStrategy, StrategyAction


def test_under_budget_returns_no_change():
    spec = ConstraintSpec(energy_budget_kwh=100)
    planner = ConstraintPlanner(spec, expected_total_steps=100)
    state = RunState(current_strategy=ExecutionStrategy(batch_size=32))
    state.cumulative_energy_kwh = 5
    state.step_count = 50  # projected: 5 * (100/50) = 10 -> 10% of budget

    decision = planner.decide(state)
    assert decision.action == StrategyAction.NO_CHANGE


def test_warning_threshold_triggers_precision_reduction():
    spec = ConstraintSpec(energy_budget_kwh=100)
    planner = ConstraintPlanner(spec, expected_total_steps=100)
    state = RunState(current_strategy=ExecutionStrategy(batch_size=32))
    state.cumulative_energy_kwh = 45
    state.step_count = 50  # projected: 90 -> 90% of budget, over 85% warning threshold

    decision = planner.decide(state)
    assert decision.action == StrategyAction.REDUCE_PRECISION
    assert decision.resulting_strategy.precision == "fp16"


def test_ladder_progression_through_all_levers():
    spec = ConstraintSpec(energy_budget_kwh=100)
    planner = ConstraintPlanner(spec, expected_total_steps=100, evaluate_every_n_steps=10)
    state = RunState(current_strategy=ExecutionStrategy(batch_size=10))
    state.cumulative_energy_kwh = 45
    state.step_count = 10  # 90% of budget, stays constant across evaluations

    decision = planner.decide(state)
    assert decision.action == StrategyAction.REDUCE_PRECISION
    state.current_strategy = decision.resulting_strategy

    decision = planner.decide(state)
    assert decision.action == StrategyAction.ENABLE_CHECKPOINTING
    state.current_strategy = decision.resulting_strategy

    # Batch size should grow, capped at 4x the initial batch size (10 -> 40).
    while state.current_strategy.batch_size < 40:
        decision = planner.decide(state)
        assert decision.action == StrategyAction.INCREASE_BATCH_SIZE
        assert decision.resulting_strategy.batch_size <= 40
        state.current_strategy = decision.resulting_strategy

    decision = planner.decide(state)
    assert decision.action == StrategyAction.INCREASE_GRAD_ACCUMULATION
    assert decision.resulting_strategy.grad_accumulation_steps == 2


def test_hysteresis_delays_early_stop_by_one_evaluation():
    spec = ConstraintSpec(energy_budget_kwh=100)
    planner = ConstraintPlanner(spec, expected_total_steps=100, evaluate_every_n_steps=10)
    planner._initial_batch_size = 10  # simulate a run that already grew from batch_size=10
    exhausted = ExecutionStrategy(
        precision="fp16", batch_size=40, grad_accumulation_steps=8, activation_checkpointing=True
    )
    state = RunState(current_strategy=exhausted)
    state.cumulative_energy_kwh = 45
    state.step_count = 50  # 90% of budget -- warning, not critical

    first = planner.decide(state)
    assert first.action == StrategyAction.NO_CHANGE

    second = planner.decide(state)
    assert second.action == StrategyAction.EARLY_STOP


def test_critical_overage_bypasses_hysteresis():
    spec = ConstraintSpec(energy_budget_kwh=100)
    planner = ConstraintPlanner(spec, expected_total_steps=100)
    planner._initial_batch_size = 10
    exhausted = ExecutionStrategy(
        precision="fp16", batch_size=40, grad_accumulation_steps=8, activation_checkpointing=True
    )
    state = RunState(current_strategy=exhausted)
    state.cumulative_energy_kwh = 110
    state.step_count = 100  # already over budget

    decision = planner.decide(state)
    assert decision.action == StrategyAction.EARLY_STOP


def test_accuracy_guard_disallows_precision_reduction():
    spec = ConstraintSpec(energy_budget_kwh=100, max_accuracy_loss=0.01)
    planner = ConstraintPlanner(spec, expected_total_steps=100, accuracy_guard_buffer=0.8)
    state = RunState(current_strategy=ExecutionStrategy(batch_size=32))
    state.cumulative_energy_kwh = 45
    state.step_count = 50
    state.baseline_val_accuracy = 0.90
    state.current_val_accuracy = 0.89  # loss=0.01 >= 0.01 * 0.8 -> guard trips

    decision = planner.decide(state)
    assert decision.action == StrategyAction.ENABLE_CHECKPOINTING


def test_no_budgets_configured_is_no_change():
    spec = ConstraintSpec()
    planner = ConstraintPlanner(spec)
    state = RunState()
    decision = planner.decide(state)
    assert decision.action == StrategyAction.NO_CHANGE
