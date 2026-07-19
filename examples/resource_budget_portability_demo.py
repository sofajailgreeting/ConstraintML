"""Demonstrates ConstraintML's actual differentiator: resource-budget portability.

`examples/carbon_savings_demo.py` shows that stopping once a validation curve has
plateaued saves real energy/carbon for negligible accuracy cost. But that demo's
`deadline` is still a fixed wall-clock cutoff, calibrated after the fact -- for that ONE
model on that ONE machine, a hardcoded epoch count picked by eyeballing the curve could
do about as well. That raises a critique from advanced users:
"Why not just pick fewer epochs?"

What a hardcoded epoch count will never do is generalize: it has no relationship to
actual compute cost, so the same epoch count means wildly different real resource usage
depending on model size or hardware speed. This demo makes that concrete with a
four-way comparison: {small model, large model} x {naive fixed-epoch count, ConstraintML
`deadline`}.

  1. A user tunes `naive_epochs` once, against a small/fast model, until the run
     comfortably fits their real time budget (calibrated here from the small model's own
     measured elapsed time -- standing in for "however the user actually arrived at that
     number").
  2. Reusing that SAME epoch count, unmodified, on a much larger/slower model blows the
     same real time budget by roughly the model's slowdown factor -- because an epoch
     count carries no information about actual compute cost.
  3. Applying a ConstraintML `deadline` (a real resource constraint, expressed in the
     unit the user actually cares about -- wall-clock time here, though the same applies
     to `energy_budget`/`carbon_budget`) to the same larger model instead respects the
     budget automatically, using however many epochs that model's actual speed allows.
     No re-tuning, no knowledge of the model's speed is required in advance.

Run with: python examples/resource_budget_portability_demo.py
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from constraintml import GreenTrainer, TrainingReport
from constraintml.optimization.telemetry.simulated import SimulatedTelemetryCollector


def build_task(seed: int = 0, n_train: int = 2000, n_val: int = 400, input_dim: int = 30, batch_size: int = 32):
    """The shared training task -- independent of student model size, so the small and
    large models below are compared on identical data."""
    torch.manual_seed(seed)
    w_true = torch.randn(input_dim, 1)

    def make_targets(x: torch.Tensor) -> torch.Tensor:
        signal = x @ w_true
        return signal + 0.3 * torch.sin(signal) + 0.4 * torch.randn(x.shape[0], 1)

    x_train = torch.randn(n_train, input_dim)
    y_train = make_targets(x_train)
    x_val = torch.randn(n_val, input_dim)
    y_val = make_targets(x_val)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, nn.MSELoss()


def build_model_factory(input_dim: int, hidden: int, seed: int = 1):
    """Returns a make_model() closure producing fresh models of a given width, all with
    identical initial weights -- so the naive and constrained runs at one model size
    start from the same point."""
    torch.manual_seed(seed)
    initial_state = nn.Sequential(
        nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
    ).state_dict()

    def make_model() -> nn.Module:
        model = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        model.load_state_dict(initial_state)
        return model

    return make_model


def run_fixed_epoch_count(
    make_model,
    train_loader,
    val_loader,
    loss_fn,
    epochs: int,
    power_watts: float = 250.0,
    evaluate_every_n_steps: int = 5,
    lr: float = 2e-4,
) -> TrainingReport:
    """The naive approach: a hardcoded epoch count, no resource awareness."""
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = GreenTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        max_epochs=epochs,
        telemetry=SimulatedTelemetryCollector(assumed_power_watts=power_watts),
        evaluate_every_n_steps=evaluate_every_n_steps,
    )
    return trainer.train()


def run_deadline_budget(
    make_model,
    train_loader,
    val_loader,
    loss_fn,
    deadline_seconds: float,
    max_epochs_cap: int = 500,
    power_watts: float = 250.0,
    evaluate_every_n_steps: int = 5,
    lr: float = 2e-4,
) -> TrainingReport:
    """The ConstraintML approach: a real wall-clock budget. Epoch count is whatever that
    budget allows -- the caller never has to know or guess it in advance."""
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = GreenTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        max_epochs=max_epochs_cap,
        deadline=deadline_seconds,
        # Loose backstop, not a literal percentage - see carbon_savings_demo.py's note
        # on the -avg_val_loss accuracy proxy.
        max_accuracy_loss=0.2,
        telemetry=SimulatedTelemetryCollector(assumed_power_watts=power_watts),
        evaluate_every_n_steps=evaluate_every_n_steps,
    )
    return trainer.train()


def run_portability_experiment(
    input_dim: int = 30,
    small_hidden: int = 32,
    large_hidden: int = 768,
    naive_epochs: int = 30,
    verbose: bool = True,
) -> dict:
    """Runs the four-way comparison and returns the reports plus the calibrated budget."""
    train_loader, val_loader, loss_fn = build_task(input_dim=input_dim)
    make_small = build_model_factory(input_dim, small_hidden)
    make_large = build_model_factory(input_dim, large_hidden)

    if verbose:
        print(f"Running small model for a fixed {naive_epochs} epochs (no budget)...")
    small_naive = run_fixed_epoch_count(make_small, train_loader, val_loader, loss_fn, epochs=naive_epochs)
    # Stand-in for "however the user actually arrived at naive_epochs": treat the time
    # it happened to take on the small model as their real, intended time budget.
    target_seconds = small_naive.elapsed_seconds
    if verbose:
        print(f"  -> took {target_seconds:.3f}s. Treating this as the real time budget.")

        print(f"Running large model for the SAME fixed {naive_epochs} epochs (no budget)...")
    large_naive = run_fixed_epoch_count(make_large, train_loader, val_loader, loss_fn, epochs=naive_epochs)
    if verbose:
        print(
            f"  -> took {large_naive.elapsed_seconds:.3f}s "
            f"({large_naive.elapsed_seconds / target_seconds:.1f}x the budget)."
        )

        print(f"Running small model with a ConstraintML deadline of {target_seconds:.3f}s...")
    small_constrained = run_deadline_budget(
        make_small, train_loader, val_loader, loss_fn, deadline_seconds=target_seconds
    )

    if verbose:
        print(f"Running large model with the SAME ConstraintML deadline of {target_seconds:.3f}s...")
    large_constrained = run_deadline_budget(
        make_large, train_loader, val_loader, loss_fn, deadline_seconds=target_seconds
    )

    return {
        "target_seconds": target_seconds,
        "small_naive": small_naive,
        "large_naive": large_naive,
        "small_constrained": small_constrained,
        "large_constrained": large_constrained,
    }


def print_portability_comparison(results: dict) -> None:
    target = results["target_seconds"]

    def row(label: str, report: TrainingReport) -> None:
        overshoot = report.elapsed_seconds / target if target > 0 else float("inf")
        print(f"{label:36}{report.elapsed_seconds:>9.3f}s{overshoot:>9.2f}x{report.epochs_completed:>10} epochs")

    print("=" * 76)
    print(f"Target time budget (from small model's naive run): {target:.3f}s")
    print("-" * 76)
    print(f"{'':36}{'elapsed':>10}{'x budget':>9}{'epochs':>16}")
    row("Small model, naive fixed-epoch", results["small_naive"])
    row("Small model, ConstraintML deadline", results["small_constrained"])
    row("Large model, naive fixed-epoch", results["large_naive"])
    row("Large model, ConstraintML deadline", results["large_constrained"])
    print("-" * 76)
    naive_overshoot = results["large_naive"].elapsed_seconds / target
    constrained_overshoot = results["large_constrained"].elapsed_seconds / target
    print(
        f"On the large model, reusing the same epoch count overshoots the real budget by "
        f"{naive_overshoot:.1f}x. The SAME `deadline` constraint instead lands at "
        f"{constrained_overshoot:.1f}x -- automatically, with no re-tuning."
    )
    print("=" * 76)


def plot_portability_comparison(
    results: dict, output_path: str = "examples/resource_budget_portability_demo.png"
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('Install matplotlib to enable charts: pip install -e ".[viz]"')
        return False

    labels = ["Small model", "Large model"]
    naive_times = [results["small_naive"].elapsed_seconds, results["large_naive"].elapsed_seconds]
    constrained_times = [results["small_constrained"].elapsed_seconds, results["large_constrained"].elapsed_seconds]
    target = results["target_seconds"]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = list(range(len(labels)))
    width = 0.35
    ax.bar([i - width / 2 for i in x], naive_times, width, label="Naive fixed-epoch count")
    ax.bar([i + width / 2 for i in x], constrained_times, width, label="ConstraintML deadline")
    ax.axhline(target, color="gray", linestyle="--", label="Target time budget")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Elapsed time (s)")
    ax.set_title("Same epoch count vs. same real budget, across model sizes")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved comparison chart to {output_path}")
    return True


def main():
    results = run_portability_experiment()
    print_portability_comparison(results)
    plot_portability_comparison(results)


if __name__ == "__main__":
    main()
