"""Auto-detection of the best available telemetry collector."""

from __future__ import annotations

import logging

from .base import TelemetryCollector
from .simulated import SimulatedTelemetryCollector

logger = logging.getLogger(__name__)


def get_telemetry_collector(prefer: str | None = None) -> TelemetryCollector:
    """Return the best available TelemetryCollector.

    Probes for a working NVML installation and falls back to simulated telemetry
    if no NVIDIA GPU / pynvml is available. Pass prefer="simulated" to force the
    fallback (useful in tests).
    """
    if prefer == "simulated":
        return SimulatedTelemetryCollector()

    try:
        import pynvml

        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except Exception:
        logger.warning("NVML/pynvml unavailable -- falling back to simulated telemetry.")
        return SimulatedTelemetryCollector()

    from .nvml import NVMLTelemetryCollector

    return NVMLTelemetryCollector()
