"""ConstraintTrainer: the shared base for GreenTrainer and SmartTrainer.

This is the thin, user-facing API layer. All adaptive behavior lives within
`constraintml.optimization` -- this class simply wires a PyTorch training loop up
to the RuntimeOptimizationEngine and reports what happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number

from ..constraints.spec import ConstraintSpec
from ..optimization.backends.pytorch_backend import PyTorchBackend
from ..optimization.carbon import CarbonModel
from ..optimization.engine import RuntimeOptimizationEngine
from ..optimization.planner import ConstraintPlanner
from ..optimization.state import EpochSnapshot, RunState
from ..optimization.strategies import ExecutionStrategy
from ..optimization.telemetry.base import TelemetryCollector
from ..optimization.telemetry.factory import get_telemetry_collector


@dataclass(frozen=True)
class TrainingReport:
    epochs_completed: int
    steps_completed: int
    energy_kwh: float
    carbon_kgco2e: float
    cost_usd: float
    elapsed_seconds: float
    stopped_early: bool
    stop_reason: str | None
    final_strategy: ExecutionStrategy
    strategy_history: list[tuple[int, ExecutionStrategy]]
    epoch_history: list[EpochSnapshot]
    baseline_val_accuracy: float | None
    final_val_accuracy: float | None

    @classmethod
    def from_state(cls, state: RunState) -> "TrainingReport":
        return cls(
            epochs_completed=state.epoch + 1 if state.step_count > 0 else 0,
            steps_completed=state.step_count,
            energy_kwh=state.cumulative_energy_kwh,
            carbon_kgco2e=state.cumulative_carbon_kgco2e,
            cost_usd=state.cumulative_cost_usd,
            elapsed_seconds=state.elapsed_seconds,
            stopped_early=state.stopped_early,
            stop_reason=state.stop_reason,
            final_strategy=state.current_strategy,
            strategy_history=list(state.strategy_history),
            epoch_history=list(state.epoch_history),
            baseline_val_accuracy=state.baseline_val_accuracy,
            final_val_accuracy=state.current_val_accuracy,
        )


class ConstraintTrainer:
    def __init__(
        self,
        model=None,
        optimizer=None,
        constraints: dict | None = None,
        *,
        energy_budget: str | Number | None = None,
        carbon_budget: str | Number | None = None,
        deadline: str | Number | None = None,
        max_accuracy_loss: str | Number | None = None,
        cost_budget: str | Number | None = None,
        optimize_for: str = "accuracy",
        train_loader=None,
        val_loader=None,
        loss_fn=None,
        max_epochs: int = 1,
        device=None,
        telemetry: TelemetryCollector | None = None,
        carbon_intensity_kg_per_kwh: float | None = None,
        cost_per_kwh_usd: float | None = None,
        planner: ConstraintPlanner | None = None,
        backend=None,
        evaluate_every_n_steps: int = 50,
    ):
        dict_spec = (
            ConstraintSpec.from_dict(constraints, optimize_for=optimize_for)
            if constraints
            else ConstraintSpec(optimize_for=optimize_for)
        )
        kwargs_spec = ConstraintSpec.from_kwargs(
            energy_budget=energy_budget,
            carbon_budget=carbon_budget,
            deadline=deadline,
            max_accuracy_loss=max_accuracy_loss,
            cost_budget=cost_budget,
            optimize_for=optimize_for,
        )
        self.spec = dict_spec.merge(kwargs_spec)

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.max_epochs = max_epochs
        self.device = device
        self.evaluate_every_n_steps = evaluate_every_n_steps

        self._telemetry = telemetry
        self._carbon_model = (
            CarbonModel(intensity_kg_per_kwh=carbon_intensity_kg_per_kwh)
            if carbon_intensity_kg_per_kwh is not None
            else CarbonModel()
        )
        self._cost_per_kwh_usd = cost_per_kwh_usd
        self._planner = planner
        self._backend = backend

    def attach(
        self, model=None, optimizer=None, train_loader=None, val_loader=None, loss_fn=None
    ) -> "ConstraintTrainer":
        """Bind model/optimizer/data after construction.

        Needed because e.g. `SmartTrainer(optimize_for="carbon", carbon_budget="8 kgCO2e")`
        is a valid README usage with no model/optimizer supplied up front.
        """
        if model is not None:
            self.model = model
        if optimizer is not None:
            self.optimizer = optimizer
        if train_loader is not None:
            self.train_loader = train_loader
        if val_loader is not None:
            self.val_loader = val_loader
        if loss_fn is not None:
            self.loss_fn = loss_fn
        return self

    def train(self) -> TrainingReport:
        if self.model is None or self.optimizer is None or self.train_loader is None:
            raise RuntimeError(
                "ConstraintTrainer requires model, optimizer, and train_loader to be set "
                "(via the constructor or .attach()) before calling train()."
            )

        backend = self._backend or PyTorchBackend(
            model=self.model,
            optimizer=self.optimizer,
            train_loader=self.train_loader,
            loss_fn=self.loss_fn,
            device=self.device,
        )
        owns_telemetry = self._telemetry is None
        telemetry = self._telemetry or get_telemetry_collector()

        expected_total_steps = self._estimate_total_steps(backend.train_loader)
        planner = self._planner or ConstraintPlanner(
            self.spec,
            evaluate_every_n_steps=self.evaluate_every_n_steps,
            expected_total_steps=expected_total_steps,
        )

        state = RunState(current_strategy=ExecutionStrategy(batch_size=backend.strategy.batch_size))
        engine = RuntimeOptimizationEngine(
            planner=planner,
            telemetry=telemetry,
            backend=backend,
            carbon_model=self._carbon_model,
            cost_per_kwh_usd=self._cost_per_kwh_usd,
            state=state,
        )

        try:
            for epoch in range(self.max_epochs):
                state.epoch = epoch
                for batch in backend.train_loader:
                    engine.begin_step()
                    loss = backend.training_step(batch, self.loss_fn)
                    engine.end_step(loss=loss)
                    if engine.should_stop():
                        break

                if self.val_loader is not None:
                    accuracy = backend.evaluate(self.val_loader)
                    state.current_val_accuracy = accuracy
                    if state.baseline_val_accuracy is None:
                        state.baseline_val_accuracy = accuracy

                state.epoch_history.append(
                    EpochSnapshot(
                        epoch=epoch,
                        cumulative_energy_kwh=state.cumulative_energy_kwh,
                        cumulative_carbon_kgco2e=state.cumulative_carbon_kgco2e,
                        elapsed_seconds=state.elapsed_seconds,
                        val_accuracy=state.current_val_accuracy,
                    )
                )

                if engine.should_stop():
                    break
        finally:
            if owns_telemetry:
                telemetry.close()

        return TrainingReport.from_state(state)

    def _estimate_total_steps(self, train_loader) -> int | None:
        try:
            steps_per_epoch = len(train_loader)
        except TypeError:
            return None
        return steps_per_epoch * self.max_epochs
