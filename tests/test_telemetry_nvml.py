import sys
import types
from unittest.mock import MagicMock

import pytest

from constraintml.optimization.telemetry.factory import get_telemetry_collector
from constraintml.optimization.telemetry.simulated import SimulatedTelemetryCollector


def _make_fake_pynvml():
    fake = types.ModuleType("pynvml")
    fake.NVML_TEMPERATURE_GPU = 0
    fake.nvmlInit = MagicMock()
    fake.nvmlShutdown = MagicMock()
    fake.nvmlDeviceGetHandleByIndex = MagicMock(return_value="handle")
    fake.nvmlDeviceGetPowerUsage = MagicMock(return_value=150_000)  # milliwatts
    fake.nvmlDeviceGetUtilizationRates = MagicMock(return_value=MagicMock(gpu=42.0, memory=17.0))
    fake.nvmlDeviceGetTemperature = MagicMock(return_value=65.0)
    return fake


def test_nvml_collector_parses_sample(monkeypatch):
    fake_pynvml = _make_fake_pynvml()
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)

    from constraintml.optimization.telemetry.nvml import NVMLTelemetryCollector

    collector = NVMLTelemetryCollector()
    sample = collector.sample()

    assert sample.power_watts == pytest.approx(150.0)
    assert sample.gpu_utilization_pct == pytest.approx(42.0)
    assert sample.memory_utilization_pct == pytest.approx(17.0)
    assert sample.temperature_c == pytest.approx(65.0)
    assert sample.is_simulated is False

    collector.close()
    fake_pynvml.nvmlShutdown.assert_called_once()


def test_factory_uses_nvml_when_available(monkeypatch):
    fake_pynvml = _make_fake_pynvml()
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)

    from constraintml.optimization.telemetry.nvml import NVMLTelemetryCollector

    collector = get_telemetry_collector()
    assert isinstance(collector, NVMLTelemetryCollector)


def test_factory_falls_back_to_simulated_when_pynvml_missing(monkeypatch):
    # Setting sys.modules["pynvml"] = None forces `import pynvml` to raise ImportError.
    monkeypatch.setitem(sys.modules, "pynvml", None)
    collector = get_telemetry_collector()
    assert isinstance(collector, SimulatedTelemetryCollector)


def test_factory_prefer_simulated_forces_fallback():
    collector = get_telemetry_collector(prefer="simulated")
    assert isinstance(collector, SimulatedTelemetryCollector)


def _real_nvml_available() -> bool:
    try:
        import pynvml

        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _real_nvml_available(), reason="Requires a real NVIDIA GPU with NVML available")
def test_real_nvml_smoke():
    from constraintml.optimization.telemetry.nvml import NVMLTelemetryCollector

    collector = NVMLTelemetryCollector()
    try:
        sample = collector.sample()
        assert sample.power_watts is not None
    finally:
        collector.close()
