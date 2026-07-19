"""Extension point for optional, provider-specific cloud adapters (AWS/Azure/GCP).

For v2+ - not implemented in v1 - ConstraintML is local-first and must work fully without
any cloud adapter. This module only documents the shape a future adapter would take
so that `CarbonModel.source` and cost accounting have a defined place to plug into.
No network or authentication code lives here.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CloudAdapter(Protocol):
    """Shape a future cloud adapter would implement. Not used by v1 -- the
    RuntimeOptimizationEngine never depends on this Protocol being satisfied.
    """

    def get_carbon_intensity(self, region: str) -> float:
        """Return regional grid carbon intensity in kg CO2e/kWh."""
        ...

    def get_energy_price(self, region: str) -> float:
        """Return regional electricity price in USD/kWh."""
        ...


__all__ = ["CloudAdapter"]
