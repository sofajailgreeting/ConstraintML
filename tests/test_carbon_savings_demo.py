import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from carbon_savings_demo import (  # noqa: E402
    build_dataset_and_model,
    plot_comparison,
    run_baseline,
    run_constrained,
)

MAX_EPOCHS = 8

# A tighter fraction than the full demo's default (0.4) gives headroom against the
# wall-clock timing noise inherent to comparing two separate, sequential process runs
# at this small test scale (see conftest.py-adjacent discussion in the demo module).
DEADLINE_FRACTION = 0.2


@pytest.fixture(scope="module")
def reports():
    train_loader, val_loader, make_model, loss_fn = build_dataset_and_model(
        n_train=800, n_val=160, input_dim=8, batch_size=16
    )
    baseline = run_baseline(
        make_model, train_loader, val_loader, loss_fn, max_epochs=MAX_EPOCHS, evaluate_every_n_steps=5
    )
    constrained = run_constrained(
        baseline,
        make_model,
        train_loader,
        val_loader,
        loss_fn,
        max_epochs=MAX_EPOCHS,
        evaluate_every_n_steps=5,
        deadline_fraction=DEADLINE_FRACTION,
    )
    return baseline, constrained


def test_baseline_runs_full_epoch_count(reports):
    baseline, _ = reports
    assert baseline.epochs_completed == MAX_EPOCHS
    assert baseline.stopped_early is False
    assert len(baseline.epoch_history) == MAX_EPOCHS


def test_constrained_uses_fewer_resources_than_baseline(reports):
    # The deadline constraint either stops the run early outright, or the planner's
    # ladder (e.g. batch-size growth) speeds up remaining epochs enough that all epochs
    # finish anyway -- either way, total steps/energy/time versus an unconstrained
    # baseline should not increase. `stopped_early` isn't asserted directly since which
    # of those two outcomes occurs depends on how far the ladder gets before the run
    # naturally finishes at this small test scale.
    baseline, constrained = reports
    assert constrained.elapsed_seconds <= baseline.elapsed_seconds
    assert constrained.energy_kwh <= baseline.energy_kwh
    assert constrained.steps_completed < baseline.steps_completed
    assert len(constrained.epoch_history) <= len(baseline.epoch_history)


def test_plot_comparison_writes_file(reports, tmp_path):
    pytest.importorskip("matplotlib")
    baseline, constrained = reports
    output_path = tmp_path / "comparison.png"
    result = plot_comparison(baseline, constrained, output_path=str(output_path))
    assert result is True
    assert output_path.exists()
