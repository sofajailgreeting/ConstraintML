"""Minimal end-to-end GreenTrainer example on synthetic data, CPU-only.

Run with: python examples/green_trainer_quickstart.py
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from constraintml import GreenTrainer


def main():
    torch.manual_seed(0)
    inputs = torch.randn(200, 10)
    targets = torch.randn(200, 1)
    dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = GreenTrainer(
        model=model,
        optimizer=optimizer,
        energy_budget="5kWh",
        carbon_budget="2kgCO2e",
        deadline="1h",
        train_loader=train_loader,
        loss_fn=nn.MSELoss(),
        max_epochs=5,
        evaluate_every_n_steps=5,
    )

    report = trainer.train()

    print(f"Steps completed:     {report.steps_completed}")
    print(f"Epochs completed:    {report.epochs_completed}")
    print(f"Energy used:         {report.energy_kwh:.6f} kWh")
    print(f"Carbon emitted:      {report.carbon_kgco2e:.6f} kg CO2e")
    print(f"Elapsed time:        {report.elapsed_seconds:.2f} s")
    print(f"Stopped early:       {report.stopped_early} ({report.stop_reason})")
    print(f"Final strategy:      {report.final_strategy}")


if __name__ == "__main__":
    main()
