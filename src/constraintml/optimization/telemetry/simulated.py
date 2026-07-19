"""Fallback telemetry collector used whenever real hardware telemetry is unavailable."""

from __future__ import annotations

import logging
import time

from .base import TelemetryCollector, TelemetrySample

logger = logging.getLogger(__name__)


class SimulatedTelemetryCollector(TelemetryCollector):
    """Returns a constant assumed power draw.

    This is the guaranteed fallback: the library must import and run on machines
    with no GPU and no NVML available (CPU-only dev boxes, CI runners). Estimates
    produced from this collector are rough and should not be treated as accurate
    measurements.
    """

    def __init__(self, assumed_power_watts: float = 250.0):
        self.assumed_power_watts = assumed_power_watts
        self._warned = False

    def sample(self) -> TelemetrySample:
        if not self._warned:
            logger.warning(
                "Using simulated telemetry (assumed constant power draw of %.1fW). "
                "Energy/carbon figures are rough estimates, not real measurements.",
                self.assumed_power_watts,
            )
            self._warned = True
        return TelemetrySample(
            timestamp=time.perf_counter(),
            power_watts=self.assumed_power_watts,
            is_simulated=True,
        )
