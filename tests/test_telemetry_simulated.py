import pytest

from constraintml.optimization.telemetry.base import TelemetrySample
from constraintml.optimization.telemetry.simulated import SimulatedTelemetryCollector


def test_simulated_sample_is_marked_simulated():
    collector = SimulatedTelemetryCollector(assumed_power_watts=100)
    sample = collector.sample()
    assert sample.is_simulated is True
    assert sample.power_watts == pytest.approx(100)


def test_trapezoidal_energy_integration():
    collector = SimulatedTelemetryCollector(assumed_power_watts=360)
    start = TelemetrySample(timestamp=0.0, power_watts=360)
    end = TelemetrySample(timestamp=10.0, power_watts=360)
    # 360W for 10 seconds = 3600 Ws = 1 Wh = 0.001 kWh
    energy_kwh = collector.integrate_energy(start, end, elapsed_seconds=10.0)
    assert energy_kwh == pytest.approx(0.001)


def test_integrate_energy_averages_varying_power():
    collector = SimulatedTelemetryCollector()
    start = TelemetrySample(timestamp=0.0, power_watts=100)
    end = TelemetrySample(timestamp=3600.0, power_watts=300)
    # avg 200W for 1 hour = 200 Wh = 0.2 kWh
    energy_kwh = collector.integrate_energy(start, end, elapsed_seconds=3600.0)
    assert energy_kwh == pytest.approx(0.2)


def test_integrate_energy_handles_missing_power():
    collector = SimulatedTelemetryCollector()
    start = TelemetrySample(timestamp=0.0, power_watts=None)
    end = TelemetrySample(timestamp=1.0, power_watts=100)
    assert collector.integrate_energy(start, end, elapsed_seconds=1.0) == 0.0
