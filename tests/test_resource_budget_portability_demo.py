import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from resource_budget_portability_demo import (  # noqa: E402
    plot_portability_comparison,
    run_portability_experiment,
)


@pytest.fixture(scope="module")
def results():
    return run_portability_experiment(
        input_dim=15,
        small_hidden=24,
        large_hidden=384,
        naive_epochs=15,
        verbose=False,
    )


def test_naive_epoch_count_is_reused_verbatim(results):
    # The whole point of the "naive" condition: the epoch count doesn't change with
    # model size, so its real-world cost isn't controlled at all.
    assert results["small_naive"].epochs_completed == 15
    assert results["large_naive"].epochs_completed == 15


def test_naive_fixed_epochs_blows_the_budget_on_the_large_model(results):
    target = results["target_seconds"]
    naive_overshoot = results["large_naive"].elapsed_seconds / target
    assert naive_overshoot > 1.3


def test_deadline_keeps_large_model_closer_to_budget_than_naive(results):
    # The core claim: reusing the same real resource constraint (not the same epoch
    # count) across model sizes lands closer to the intended budget.
    large_naive = results["large_naive"]
    large_constrained = results["large_constrained"]
    assert large_constrained.elapsed_seconds < large_naive.elapsed_seconds
    assert large_constrained.epochs_completed < large_naive.epochs_completed


def test_small_model_is_unaffected_either_way(results):
    # For the model the budget was implicitly tuned against, naive and constrained
    # should land in the same ballpark -- the difference only shows up once the model
    # changes.
    target = results["target_seconds"]
    small_constrained = results["small_constrained"]
    assert small_constrained.elapsed_seconds / target < 1.5


def test_plot_portability_comparison_writes_file(results, tmp_path):
    pytest.importorskip("matplotlib")
    output_path = tmp_path / "portability.png"
    result = plot_portability_comparison(results, output_path=str(output_path))
    assert result is True
    assert output_path.exists()
