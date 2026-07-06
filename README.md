# ConstraintML
A simple layer that wraps existing training frameworks to reduce long-term energy costs and support carbon accounting efforts. ConstraintML looks to create a shift from today's configuration-driven ML tooling toward constraint-driven ML infrastructure, enabling organizations to easily configure production flows that fit into various budgetary and compliance requirements.

## Getting Started
Run the following command in your environment:
pip install git+https://github.com/sofajailgreeting/ConstraintML

Before training your model, import a relevant class like GreenTrainer or SmartTrainer and run:

--------------------------
trainer = GreenTrainer(
    model=model,
    optimizer=optimizer,
    energy_budget=100,          # kWh
    carbon_budget=25,           # kg CO₂e
    deadline="8h",
)

trainer.train()

-------------------------

trainer = SmartTrainer(
    optimize_for="carbon",
    carbon_budget="8 kgCO2e",
)

trainer.train()

------------------------

## Architecture

Core Optimizing Functions
Cloud Native APIs
Optional Reporting Visualizations
