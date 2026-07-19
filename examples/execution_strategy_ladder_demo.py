"""Demonstrates the ConstraintPlanner's execution-strategy ladder end-to-end.

`carbon_savings_demo.py` and `resource_budget_portability_demo.py` both show savings
that come from a single lever: a `deadline` that eventually triggers early stop. Neither
exercises the ladder ConstraintPlanner actually walks through *before* it ever considers
stopping: reduce precision -> enable activation checkpointing -> grow batch size ->
increase gradient accumulation (see `optimization/planner.py`'s `_next_lever`). That
sequence is covered by a unit test (`test_ladder_progression_through_all_levers`)
against a hand-built `RunState`, but never against a real `PyTorchBackend` doing real
forward/backward passes.

This demo forces the full ladder deliberately: it measures one unconstrained epoch's
real energy use, then re-runs training with `energy_budget` set to a tiny fraction of
that measured value (2%) -- small enough that the run is "over budget" from the very
first planner evaluation and stays that way regardless of which lever is currently
active, so every evaluation applies the next untried lever until none remain, then
early-stops. Precision changes (real `torch.autocast`), activation checkpointing (real
`checkpoint_sequential`), and gradient accumulation (real micro-step counting) all take
effect on the very next training step. Batch size is the one lever that does NOT: the
outer training loop iterates a DataLoader object captured at the start of each epoch, so
a mid-epoch batch resize (`PyTorchBackend._resize_batch` builds a *new* DataLoader) only
takes effect once the *next* epoch starts iterating `backend.train_loader` again -- a
real, load-bearing detail of the current loop structure, not a bug in this demo. Enough
epochs are configured here for that to happen at least once, so the resized batch is
actually exercised, not just recorded.

Caveat this demo does NOT claim: it does not show these levers reducing energy on THIS
hardware. `SimulatedTelemetryCollector` assumes constant wattage, so measured energy
here tracks real wall-clock time; per CLAUDE.md's CPU autocast note, precision reduction
is not a guaranteed win on a CPU with no low-precision acceleration hardware -- its
payoff is real but hardware-dependent (expected mainly on CUDA with tensor cores). The
point of this demo is narrower: showing the planner actually walks the full ladder, in
order, against real training code, before it ever early-stops -- not that doing so is
fast on a laptop CPU.

Run with: python examples/execution_strategy_ladder_demo.py
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from constraintml import GreenTrainer, TrainingReport
from constraintml.optimization.strategies import ExecutionStrategy
from constraintml.optimization.telemetry.simulated import SimulatedTelemetryCollector


def build_dataset_and_model(seed: int = 0, n_train: int = 300, input_dim: int = 20, batch_size: int = 10):
    """A plain regression task -- this demo is about the strategy trace, not accuracy,
    so the target function doesn't need the gradual-convergence shape the savings demos
    use. `nn.Sequential` is required for activation checkpointing to actually engage
    (see `PyTorchBackend._forward`)."""
    torch.manual_seed(seed)
    inputs = torch.randn(n_train, input_dim)
    targets = torch.randn(n_train, 1)
    train_loader = DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=True)

    initial_state = nn.Sequential(
        nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
    ).state_dict()

    def make_model() -> nn.Module:
        model = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        model.load_state_dict(initial_state)
        return model

    return train_loader, make_model, nn.MSELoss()


def measure_unconstrained_epoch_energy(
    make_model, train_loader, loss_fn, power_watts: float = 250.0, lr: float = 2e-4
) -> float:
    """One unconstrained epoch, used purely to calibrate a deliberately-too-tight budget
    for `run_ladder` below -- mirrors how `carbon_savings_demo.py` calibrates `deadline`
    from a baseline's measured elapsed time."""
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = GreenTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        loss_fn=loss_fn,
        max_epochs=1,
        telemetry=SimulatedTelemetryCollector(assumed_power_watts=power_watts),
    )
    return trainer.train().energy_kwh


def run_ladder(
    make_model,
    train_loader,
    loss_fn,
    energy_budget_kwh: float,
    max_epochs: int = 10,
    power_watts: float = 250.0,
    evaluate_every_n_steps: int = 5,
    lr: float = 2e-4,
) -> TrainingReport:
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = GreenTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        loss_fn=loss_fn,
        max_epochs=max_epochs,
        energy_budget=energy_budget_kwh,
        telemetry=SimulatedTelemetryCollector(assumed_power_watts=power_watts),
        evaluate_every_n_steps=evaluate_every_n_steps,
    )
    return trainer.train()


def describe_change(before: ExecutionStrategy, after: ExecutionStrategy) -> str:
    if before.precision != after.precision:
        return f"REDUCE_PRECISION       -> precision={after.precision}"
    if before.activation_checkpointing != after.activation_checkpointing:
        return "ENABLE_CHECKPOINTING   -> activation_checkpointing=True"
    if before.batch_size != after.batch_size:
        return f"INCREASE_BATCH_SIZE    -> batch_size={after.batch_size}"
    if before.grad_accumulation_steps != after.grad_accumulation_steps:
        return f"INCREASE_GRAD_ACCUM    -> grad_accumulation_steps={after.grad_accumulation_steps}"
    return "(unrecognized strategy change)"


def print_ladder_trace(report: TrainingReport, initial_batch_size: int) -> None:
    print("=" * 72)
    print("Execution-strategy ladder trace")
    print("-" * 72)
    previous = ExecutionStrategy(batch_size=initial_batch_size)
    for step, strategy in report.strategy_history:
        print(f"  step {step:>5}:  {describe_change(previous, strategy)}")
        previous = strategy
    print("-" * 72)
    print(f"Final strategy:   {report.final_strategy}")
    print(f"Stopped early:    {report.stopped_early}")
    if report.stop_reason:
        print(f"Stop reason:      {report.stop_reason}")
    print(f"Steps completed:  {report.steps_completed}")
    print(f"Epochs completed: {report.epochs_completed}")
    print("=" * 72)


def main():
    train_loader, make_model, loss_fn = build_dataset_and_model()

    print("Measuring one unconstrained epoch's real energy use...")
    baseline_energy_kwh = measure_unconstrained_epoch_energy(make_model, train_loader, loss_fn)
    print(f"  -> {baseline_energy_kwh:.8f} kWh over a full epoch.")

    # Deliberately tiny: ~2% of one real epoch's energy, so the run reads as "over
    # budget" from the very first evaluation and stays that way through the whole
    # ladder -- see the module docstring for why that's the point of this demo.
    tight_budget = baseline_energy_kwh * 0.02
    print(f"Re-running with energy_budget={tight_budget:.8f} kWh (2% of that) to force the full ladder...")
    report = run_ladder(make_model, train_loader, loss_fn, energy_budget_kwh=tight_budget)

    print_ladder_trace(report, initial_batch_size=train_loader.batch_size)


if __name__ == "__main__":
    main()
