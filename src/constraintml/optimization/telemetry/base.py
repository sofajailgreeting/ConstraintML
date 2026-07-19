"""Telemetry collection interface shared by all hardware backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class TelemetrySample:
    timestamp: float
    power_watts: float | None
    gpu_utilization_pct: float | None = None
    memory_utilization_pct: float | None = None
    temperature_c: float | None = None
    is_simulated: bool = False


class TelemetryCollector(ABC):
    """Source of point-in-time hardware telemetry samples."""

    @abstractmethod
    def sample(self) -> TelemetrySample:
        """Return a telemetry reading for the current instant."""

    def integrate_energy(
        self, start: TelemetrySample, end: TelemetrySample, elapsed_seconds: float
    ) -> float:
        """Trapezoidal integration of power over time -> kWh.

        Shared by all collectors so NVML- and simulation-backed readings are
        combined into energy estimates the same way.
        """
        if start.power_watts is None or end.power_watts is None or elapsed_seconds <= 0:
            return 0.0
        avg_watts = (start.power_watts + end.power_watts) / 2
        watt_hours = avg_watts * elapsed_seconds / 3600 # 60 seconds * 60 min
        return watt_hours / 1000

    def close(self) -> None:
        """Release any underlying resources. No-op by default."""
