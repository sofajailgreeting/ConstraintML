"""NVIDIA NVML-backed telemetry collector.

`pynvml` is imported lazily inside the constructor so the core package never
hard-depends on it: install the `constraintml[nvidia]` extra to use this collector.
"""

from __future__ import annotations

import time

from .base import TelemetryCollector, TelemetrySample


class NVMLTelemetryCollector(TelemetryCollector):
    def __init__(self, device_index: int = 0):
        import pynvml

        self._pynvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    def sample(self) -> TelemetrySample:
        pynvml = self._pynvml
        power_watts = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0

        try:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            gpu_utilization_pct = float(utilization.gpu)
            memory_utilization_pct = float(utilization.memory)
        except Exception:
            gpu_utilization_pct = None
            memory_utilization_pct = None

        try:
            temperature_c = float(
                pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
            )
        except Exception:
            temperature_c = None

        return TelemetrySample(
            timestamp=time.perf_counter(),
            power_watts=power_watts,
            gpu_utilization_pct=gpu_utilization_pct,
            memory_utilization_pct=memory_utilization_pct,
            temperature_c=temperature_c,
            is_simulated=False,
        )

    def close(self) -> None:
        self._pynvml.nvmlShutdown()
