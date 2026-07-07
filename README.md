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


At its core, this project is designed as a **constraint-aware runtime** that sits between an ML framework (e.g., PyTorch, JAX, or TensorFlow) and the underlying hardware. Rather than requiring developers to manually tune dozens of low-level training parameters, the runtime allows them to specify **high-level objectives and constraints**—such as energy budgets, carbon budgets, cost limits, runtime deadlines, or acceptable accuracy loss—and automatically adapts the training process to satisfy those constraints.

```python
trainer = ConstraintTrainer(
    model=model,
    constraints={
        "energy": "<50kWh",
        "carbon": "<10kgCO2e",
        "deadline": "8h",
        "max_accuracy_loss": "0.5%"
    }
)

trainer.train()
```

Instead of exposing configuration options such as mixed precision settings, batch size, gradient accumulation, checkpointing, optimizer selection, or early stopping as independent decisions, the runtime treats these as optimization variables. During execution, it continuously evaluates the current state of training and determines which adjustments best satisfy the user-defined constraints while maximizing model quality.

### Runtime Optimization Engine

The central component of the architecture is the **Runtime Optimization Engine**, which acts as a constraint solver for ML workloads. Rather than statically configuring a training job before execution, the optimizer continuously observes the training process and dynamically modifies execution strategies as conditions change.

Potential optimization decisions include:

* Selecting numerical precision (FP32, FP16, FP8, INT8, etc.)
* Adjusting batch size
* Modifying gradient accumulation
* Enabling or disabling activation/gradient checkpointing
* Choosing optimizer implementations
* Dynamically controlling parallelism
* Applying early stopping based on diminishing returns
* Selecting hardware-specific execution kernels
* Scheduling pauses or resumptions of training

The objective is not simply to maximize throughput, but to maximize model performance while remaining within user-defined resource constraints.

### Local-First Design

The architecture is intentionally **cloud-independent**. The runtime should function without requiring authentication with AWS, Azure, Google Cloud, or any external service. A user should be able to execute the framework on:

* a local workstation
* an on-premises GPU server
* a university HPC cluster
* a Kubernetes cluster
* any cloud provider

without changing the core optimization logic.

This design keeps the optimization engine portable and allows it to operate wherever training occurs.

### Hardware Telemetry Layer

Rather than relying primarily on cloud provider APIs, the runtime gathers telemetry directly from the underlying hardware whenever possible.

For NVIDIA GPUs, this information can be collected using NVIDIA's Management Library (NVML), while similar interfaces exist for AMD (ROCm SMI), Intel (RAPL), and operating system power management APIs.

Telemetry may include:

* instantaneous power draw
* average power consumption
* GPU utilization
* CPU utilization
* memory utilization
* memory bandwidth
* clock frequencies
* temperature
* elapsed execution time

From these measurements, the runtime can estimate total energy consumption throughout training. Since energy is simply power integrated over time, accurate estimates can often be obtained entirely from local hardware telemetry without requiring any cloud-specific integration.

### Constraint Planner

The Constraint Planner serves as the policy layer of the runtime. Rather than optimizing for a single metric such as throughput or latency, it performs multi-objective optimization across competing goals.

For example, a user may request:

* maximize validation accuracy
* while consuming less than 50 kWh
* completing within 8 hours
* and limiting accuracy degradation to 0.5%

The planner evaluates the current state of execution and selects the combination of runtime optimizations most likely to satisfy those constraints.

This represents a shift from today's configuration-driven ML workflows toward a more declarative model in which developers specify *what* they want, rather than *how* to achieve it.

### Cloud Integration (Optional)

While cloud integration is not required, the architecture is designed to support optional provider-specific adapters.

Adapters for platforms such as AWS, Azure, or Google Cloud could provide additional signals that are unavailable from local hardware alone, including:

* regional carbon intensity
* electricity pricing
* spot instance pricing
* hardware availability
* instance recommendations
* checkpoint migration opportunities

These adapters do not replace the Runtime Optimization Engine; instead, they extend its visibility into the broader execution environment.

For example, if two equivalent GPU instances are available in different regions, the planner may choose to execute training in the region with significantly lower carbon intensity or lower operating cost. Likewise, the runtime could checkpoint training and resume on a newly available accelerator if doing so better satisfies the user's objectives.

### Layered Architecture

Conceptually, the framework can be viewed as a new software layer inserted between existing ML frameworks and the underlying compute infrastructure:

```
Application
      │
Constraint-Aware Runtime
      │
ML Framework (PyTorch / JAX / TensorFlow)
      │
Hardware / Cluster / Cloud
```

The runtime remains agnostic to where computation occurs. Whether executing on a laptop GPU, an enterprise cluster, or a hyperscale cloud provider, its responsibility is the same: continuously observe execution, evaluate user-defined constraints, and adapt the training process to maximize model quality while minimizing resource consumption.

Ultimately, the long-term vision is to treat energy, carbon emissions, execution time, and monetary cost as first-class optimization objectives, elevating them alongside accuracy in modern machine learning workflows.

