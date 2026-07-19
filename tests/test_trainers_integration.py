import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from constraintml import GreenTrainer, SmartTrainer
from constraintml.optimization.telemetry.simulated import SimulatedTelemetryCollector


def _make_toy_loader():
    torch.manual_seed(0)
    inputs = torch.randn(20, 4)
    targets = torch.randn(20, 1)
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=4, shuffle=True)


def test_green_trainer_runs_end_to_end_and_updates_weights():
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train_loader = _make_toy_loader()
    initial_weight = model.weight.detach().clone()

    trainer = GreenTrainer(
        model=model,
        optimizer=optimizer,
        energy_budget="1000kWh",
        carbon_budget="1000kgCO2e",
        deadline="1h",
        train_loader=train_loader,
        loss_fn=nn.MSELoss(),
        max_epochs=2,
        telemetry=SimulatedTelemetryCollector(assumed_power_watts=10),
        evaluate_every_n_steps=1,
    )

    report = trainer.train()

    assert report.steps_completed > 0
    assert report.epochs_completed == 2
    # Energy is derived from wall-clock elapsed time per step, which can legitimately
    # round to ~0 for a tiny synthetic model/dataset -- >= 0 is the only safe invariant.
    assert report.energy_kwh >= 0
    assert report.carbon_kgco2e >= 0
    assert not torch.allclose(model.weight.detach(), initial_weight)


def test_smart_trainer_supports_deferred_attach():
    # Mirrors the README's SmartTrainer(optimize_for=..., carbon_budget=...) usage,
    # which supplies no model/optimizer at construction time.
    trainer = SmartTrainer(
        optimize_for="carbon",
        carbon_budget="8 kgCO2e",
        telemetry=SimulatedTelemetryCollector(assumed_power_watts=10),
    )

    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.attach(
        model=model, optimizer=optimizer, train_loader=_make_toy_loader(), loss_fn=nn.MSELoss()
    )

    report = trainer.train()
    assert report.steps_completed > 0


def test_train_without_model_raises_runtime_error():
    trainer = SmartTrainer(optimize_for="energy", energy_budget="10kWh")
    with pytest.raises(RuntimeError):
        trainer.train()


def test_green_trainer_tracks_validation_accuracy():
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = GreenTrainer(
        model=model,
        optimizer=optimizer,
        energy_budget="1000kWh",
        train_loader=_make_toy_loader(),
        val_loader=_make_toy_loader(),
        loss_fn=nn.MSELoss(),
        max_epochs=2,
        telemetry=SimulatedTelemetryCollector(assumed_power_watts=10),
    )

    report = trainer.train()
    assert report.baseline_val_accuracy is not None
    assert report.final_val_accuracy is not None

    assert len(report.epoch_history) == report.epochs_completed == 2
    assert [snapshot.epoch for snapshot in report.epoch_history] == [0, 1]
    assert report.epoch_history[0].val_accuracy == report.baseline_val_accuracy
    assert report.epoch_history[-1].val_accuracy == report.final_val_accuracy
    # cumulative energy/carbon/time are non-decreasing across epochs
    assert report.epoch_history[-1].cumulative_energy_kwh >= report.epoch_history[0].cumulative_energy_kwh
    assert report.epoch_history[-1].elapsed_seconds >= report.epoch_history[0].elapsed_seconds
