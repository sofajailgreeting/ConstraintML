# ConstraintML

A constraint-aware layer that wraps PyTorch training so you specify **budgets** — energy, carbon, cost, a deadline, an acceptable accuracy loss — instead of manually tuning precision, batch size, gradient accumulation, checkpointing, and early stopping by hand. A deterministic, rule-based runtime observes training as it runs and adjusts execution strategy to satisfy those budgets while preserving model quality. ConstraintML looks to create a shift from today's configuration-driven ML tooling toward constraint-driven ML infrastructure, enabling organizations to easily configure production flows that fit into various budgetary and compliance requirements.

```python
trainer = GreenTrainer(
    model=model,
    optimizer=optimizer,
    energy_budget=100, # kWh
    carbon_budget=25, # kg CO2e
    deadline="8h",
)
trainer.train()
```

## Status

v1 is implemented and working, scoped to **PyTorch only**. `ConstraintTrainer`, `GreenTrainer`, and `SmartTrainer` (`src/constraintml/`) wrap a real PyTorch training loop, backed by:

- a deterministic, rule-based `ConstraintPlanner` — explicitly a heuristic, not a learned/ML optimizer, despite the project's tooling category
- a `RuntimeOptimizationEngine` that samples telemetry every step and applies planner decisions to the backend
- hardware telemetry that uses real NVIDIA NVML power readings when available (optional `pynvml` dependency) and falls back to a simulated constant-wattage collector otherwise, so the library imports and trains on CPU-only machines with no GPU at all

62 tests pass (1 additional test is skipped unless it's run on a real NVIDIA GPU with NVML available) covering constraint parsing, the planner's decision ladder, telemetry, carbon accounting, and end-to-end trainer integration. See `examples/` below for runnable demonstrations with real output.

## Installation

```bash
pip install -e ".[dev]" # editable install + pytest
pip install -e ".[dev,nvidia]" # + real NVML GPU telemetry via pynvml
pip install -e ".[dev,viz]" # + matplotlib, for the example charts below
```

Once published:
```bash
pip install git+https://github.com/sofajailgreeting/ConstraintML
```

Requires Python 3.9+ and PyTorch 2.0+ (`torch>=2.0` is the only hard runtime dependency).

## Running

```bash
pytest                                                                     # full test suite, no GPU required
pytest tests/test_planner.py::test_ladder_progression_through_all_levers  # a single test

python examples/green_trainer_quickstart.py            # end-to-end sanity check, CPU-only
python examples/carbon_savings_demo.py                 # baseline vs. deadline-constrained run
python examples/resource_budget_portability_demo.py    # same deadline across two model sizes
python examples/execution_strategy_ladder_demo.py       # forces the full lever ladder end-to-end
```

## Usage

```python
trainer = GreenTrainer(
    model=model,
    optimizer=optimizer,
    energy_budget=100,          # kWh
    carbon_budget=25,           # kg CO2e
    deadline="8h",
)
trainer.train()
```

```python
trainer = SmartTrainer(
    optimize_for="carbon",
    carbon_budget="8 kgCO2e",
)
# model/optimizer/train_loader can be attached later via trainer.attach(...) --
# useful when not available at construction time.
trainer.train()
```

```python
trainer = ConstraintTrainer(
    model=model,
    constraints={
        "energy": "<50kWh",
        "carbon": "<10kgCO2e",
        "deadline": "8h",
        "max_accuracy_loss": "0.5%",
    },
)
trainer.train()
```

`GreenTrainer` fixes `optimize_for="accuracy"` and hides that parameter. `SmartTrainer` exposes `optimize_for` as its primary knob (used as a planner tie-breaker among competing budgets, not a different algorithm). `ConstraintTrainer` is the shared base both build on.

## v1 Architecture

Layering, thinnest/user-facing at the top:

```
Application
      │
trainers/          <- ConstraintTrainer, GreenTrainer, SmartTrainer (thin, user-facing)
      │
optimization/      <- planner, engine, telemetry, carbon accounting, backends
      │
PyTorch
```

- **`constraints/`** — parses user-facing budget strings (`"8h"`, `"<50kWh"`, `"25 kg CO2e"`, `"0.5%"`) into a typed, frozen `ConstraintSpec`. Shared by `trainers/` and `optimization/planner.py`.
- **`trainers/`** — `ConstraintTrainer` (`base.py`) merges a `constraints={...}` dict with named kwargs into one `ConstraintSpec` (named kwargs win), supports deferred `model`/`optimizer` binding via `.attach()`, and owns the training loop in `.train()`, wiring telemetry + planner + backend together via `RuntimeOptimizationEngine` and returning a `TrainingReport`.
- **`optimization/planner.py`** — `ConstraintPlanner`: every `evaluate_every_n_steps`, linearly projects cumulative energy/carbon/cost to end-of-run, compares against budgets, and — if the worst metric is trending over the 85% warning threshold — applies the next untried lever in a fixed ladder:

  **reduce precision (fp32→fp16) → enable activation checkpointing → grow batch size (capped at 4× initial) → increase gradient accumulation → early stop.**

  An accuracy-loss guard disables further precision reduction once validation accuracy has regressed too far; hysteresis requires two consecutive over-warning evaluations before early-stopping (bypassed immediately if a budget is already exceeded). The ladder is monotonic — v1 never steps back down once a lever is used.
- **`optimization/engine.py`** — `RuntimeOptimizationEngine`: per-step `begin_step()`/`end_step()` sample telemetry, integrate energy (trapezoidal power integration), update carbon/cost via `CarbonModel`, and apply any resulting `ExecutionStrategy` to the backend on planner-evaluation steps.
- **`optimization/telemetry/`** — `TelemetryCollector` interface. `NVMLTelemetryCollector` lazily imports `pynvml` for real GPU power/utilization/temperature. `SimulatedTelemetryCollector` (constant assumed wattage) is the guaranteed CPU-safe fallback. `factory.get_telemetry_collector()` auto-selects between them by probing `pynvml.nvmlInit()`.
- **`optimization/backends/pytorch_backend.py`** — `PyTorchBackend`, the only real `ExecutionBackend`: real `torch.autocast` precision switches (dtype chosen per device — `bfloat16` on CPU, `float16` on CUDA), batch-size growth by rebuilding the `DataLoader` (map-style datasets only), gradient-accumulation micro-stepping, and `checkpoint_sequential`-based activation checkpointing (`nn.Sequential` models only — logs a warning and no-ops for arbitrary models it can't safely wrap).
- **`optimization/carbon.py`** — `CarbonModel`: a static `energy_kwh * intensity_kg_per_kwh` conversion (default 0.4 kg CO2e/kWh, a rough global-average grid estimate), overridable per-trainer.
- **`cloud/`** — a docstring-only `CloudAdapter` Protocol stub. No network/auth code; not used (yet) for v1 — exists purely so `CarbonModel.source` and future regional cost/carbon lookups have a defined shape to grow into.

## Examples & Results

Four runnable scripts in `examples/`, in increasing order of what they demonstrate. The numbers below are one real run on a CPU-only dev machine (`SimulatedTelemetryCollector`, fixed random seeds) — exact energy/time figures will vary by hardware and machine load, but the qualitative story (relative proportions, which lever fires, stop reasons) is what to look at, and each script prints/plots its own numbers when you run it.

### 1. `green_trainer_quickstart.py` — minimal sanity check
Trains a tiny model on synthetic data, CPU-only, no GPU or extras required.
```
Steps completed:     65
Epochs completed:    5
Energy used:         0.000002 kWh
Carbon emitted:      0.000001 kg CO2e
Final strategy:      ExecutionStrategy(precision='fp32', batch_size=16, grad_accumulation_steps=1, activation_checkpointing=False)
```

### 2. `carbon_savings_demo.py` — deadline-triggered early stop, measured end-to-end
Baseline (no budget) vs. `GreenTrainer` with a `deadline` set to 40% of the baseline's wall time, on a task designed to plateau then mildly overfit. `energy_budget`/`carbon_budget` are also passed (with generous headroom) so they never bind — `deadline` (a live `elapsed_seconds / deadline_seconds` ratio) is the only thing that actually triggers the stop here. Energy/carbon budgets are projected by extrapolating the current per-step rate to the end of the run instead, so a tight one would fire almost immediately rather than partway through; see the script's docstring for the full explanation.
```
                                Baseline       Constrained
Epochs completed                      25                14
Energy (kWh)                    0.000073          0.000045
Carbon (kg CO2e)                0.000029          0.000018
Final val accuracy*              -0.0878           -0.0859

Energy saved:   38.3%
Carbon saved:   38.3%   (mirrors energy 1:1 -- CarbonModel is a static multiplier, not an independent measurement)
Accuracy delta: +0.0019 (constrained is actually *better* here -- stopping near the plateau avoided the overfitting tail)
```
*accuracy is `-avg validation loss`, not a task-specific accuracy metric.

**What this does and doesn't show:** the energy/carbon savings above are a real, measured side effect of stopping sooner on the clock — not evidence that ConstraintML detected an accuracy plateau or weighed carbon against accuracy to pick this stopping point. The planner has no such lookahead; `deadline` is a plain wall-clock ratio (see `_fractions` in `optimization/planner.py`). The accuracy-neutral (here, slightly better) outcome is a property of this dataset, which is deliberately constructed to plateau then mildly overfit, and of `deadline` having been calibrated after the fact from the baseline's own wall time — a hand-picked epoch count inspecting the same curve would land similarly, for this one model on this one machine. What the example *does* demonstrate honestly: the full telemetry → energy integration → carbon conversion → report pipeline runs end-to-end on real, non-mocked numbers, and a `deadline` constraint gives you a resource-anchored stand-in for a hardcoded epoch count that costs nothing here. Whether that portability actually holds *across* model size and hardware — the harder and more useful claim — is what Example 3 tests.

### 3. `resource_budget_portability_demo.py` — the actual differentiator
The same hardcoded epoch count (30) vs. the same `deadline`, applied first to a small model, then to a 24x-wider model, on identical data:
```
                                       elapsed   x budget   epochs
Small model, naive fixed-epoch          1.328s      1.00x       30
Small model, ConstraintML deadline      1.288s      0.97x       30
Large model, naive fixed-epoch          5.707s      4.30x       30
Large model, ConstraintML deadline      1.722s      1.30x       10
```
Reusing the same epoch count on the larger model blows the same real time budget by 4.3x, because an epoch count carries no information about actual compute cost. The same `deadline` constraint instead self-adjusts to 10 epochs and lands at 1.3x, automatically, with no re-tuning. **This is the real differentiator of ConstraintML, especially in v1** — not smarter plateau detection, but a resource constraint that stays meaningful across model size and hardware.

### 4. `execution_strategy_ladder_demo.py` — the full lever ladder
A deliberately tiny `energy_budget` (2% of one measured unconstrained epoch) forces `ConstraintPlanner` through every lever, against a real `PyTorchBackend`:
```
step  5:     REDUCE_PRECISION       -> precision=fp16
step 10:     ENABLE_CHECKPOINTING   -> activation_checkpointing=True
step 15-30:  INCREASE_BATCH_SIZE    -> batch_size 15 -> 22 -> 33 -> 40 (capped at 4x initial)
step 35-65:  INCREASE_GRAD_ACCUM    -> grad_accumulation_steps 2 -> 8 (capped)
Stopped early: True, after 70 steps spanning 6 real epochs
Stop reason: "energy projected at 126008% of budget; all optimization levers exhausted."
```
This does **not** claim these levers make training faster on CPU — see the script's docstring and the CPU autocast note below. It demonstrates that the ladder itself executes correctly end-to-end against real training code (real autocast, real `checkpoint_sequential`, a real `DataLoader` rebuild), not just in a unit test against a mocked planner state.

## Current Scope: What This Means for Production Teams

v1 targets a narrower use case than the long-term vision below. Concretely:

**Good fit today:**
- Single-process PyTorch training jobs (one GPU, or CPU-only) where you want automatic, soft guardrails against blowing a time, energy, carbon, or cost budget.
- Teams who currently hardcode an epoch count "that felt about right" and want that number to keep working when the model or hardware changes, without re-tuning — this is exactly what `resource_budget_portability_demo.py` demonstrates.
- Environments with no GPU at all: the library never requires a GPU or `pynvml`; `SimulatedTelemetryCollector` is a real, tested fallback, not a stub.

**Not yet a fit:**
- **Multi-GPU or distributed training** — v1 has no distributed strategy switching; the planner and backend assume a single device.
- **Regulatory- or audit-grade carbon accounting** — `CarbonModel` is a static `energy_kwh * 0.4` global-average multiplier, not a regional or real-time grid intensity figure. Treat carbon numbers as directional, not compliance-ready, until a `cloud/` adapter supplies real intensity data.
- **Cost-sensitive cloud scheduling** — there's no cloud pricing/spot integration; `cost_budget` exists in the API but has no cloud cost signal behind it today.
- **Guaranteed speedups from precision reduction** — the `REDUCE_PRECISION` lever's real-world payoff is hardware-dependent. Measured on this project's dev CPU, `torch.autocast(dtype=torch.float16)` was ~44x *slower* than fp32 (CPUs lack native fp16 compute kernels and fall back to emulation); v1 uses `bfloat16` on CPU instead to avoid that catastrophic case, but even `bfloat16` wasn't a clear win on hardware without low-precision acceleration. As such, expect real wins mainly on CUDA GPUs with tensor cores.
- **Precision "recovery"** — once the planner reduces precision or grows batch size, v1 never steps back down, even if usage later drops back under budget. This is a deliberate, monotonic simplification (to avoid oscillation), not an oversight.

## What's Out of Scope for v1 (documented extension points)

These are architectural extension points the codebase is designed to grow into — **not committed or scheduled features**:

- JAX / TensorFlow execution backends (v1 is PyTorch-only)
- Real cloud adapters (AWS / Azure / GCP) for regional carbon intensity, spot pricing, and instance recommendations — `cloud/` currently ships only a docstring-only `CloudAdapter` Protocol stub
- FP8 / INT8 execution (named in the `ExecutionStrategy` precision enum for API completeness, never selected by v1's planner or backend)
- AMD ROCm SMI / Intel RAPL telemetry (NVIDIA NVML and the CPU-fallback collector only today)
- Multi-GPU / distributed strategy switching
- Pause/resume or checkpoint migration across hardware
- Reporting/visualization dashboards beyond the `[viz]` matplotlib charts in `examples/`
- Precision "recovery" (stepping back up once back under budget)

If your use case depends on any of these, treat it as a gap to plan around today rather than an assumed near-term feature — check current project status directly before committing to a timeline around any of them.

## Long-Term Vision

Beyond v1's PyTorch-only, single-device scope, the project's direction is to treat energy, carbon, execution time, and cost as first-class optimization objectives, on equal footing with accuracy, across any ML framework and any hardware:

- **Local-first, cloud-independent**: run unmodified on a laptop, an on-prem GPU server, a university HPC cluster, a Kubernetes cluster, or any cloud provider, gathering telemetry directly from hardware (NVML, ROCm SMI, RAPL) rather than depending on cloud-provider APIs.
- **Genuine multi-objective planning**: evolve `ConstraintPlanner` from today's fixed rule-based ladder toward optimizing across competing goals simultaneously (accuracy vs. energy vs. carbon vs. deadline vs. cost), and toward hardware-specific execution kernel selection and dynamic parallelism control.
- **Optional cloud adapters**: `cloud/`'s `CloudAdapter` Protocol is meant to eventually supply regional carbon intensity, electricity pricing, spot pricing, and checkpoint-migration signals — extending the planner's visibility without replacing its core logic or requiring cloud auth for the base library to function.

This is a direction the architecture is designed to grow into, not a promise of any specific feature landing in any specific release - although most of this should be expected sooner rather than later.

## Publishing (for maintainers)

Packaging is `pyproject.toml`-only (PEP 517/518 via `setuptools.build_meta`). To build and upload a release:

```bash
python -m pip install --upgrade build twine
python -m build                        # produces dist/*.whl and dist/*.tar.gz
python -m twine check dist/*           # validates metadata/README rendering before upload
python -m twine upload dist/*          # prompts for PyPI credentials (or use a PyPI API token)
```

Bump `version` in `pyproject.toml` (and `__version__` in `src/constraintml/__init__.py`, which is not currently read from the package metadata) before each release. `MANIFEST.in` ensures `LICENSE` and `README.md` are included in the sdist.

## License

MIT — see [LICENSE](LICENSE).
