"""Carbon accounting: converts energy consumption into estimated carbon emissions."""

from __future__ import annotations

from dataclasses import dataclass

# Rough global-average grid carbon intensity. This is a coarse approximation --
# override with a locally accurate figure via `carbon_intensity_kg_per_kwh=` on any
# trainer. A future `constraintml.cloud` adapter is the intended extension point for
# sourcing a regionally accurate figure automatically (see CarbonModel.source).
DEFAULT_GRID_CARBON_INTENSITY_KG_PER_KWH = 0.4


@dataclass
class CarbonModel:
    intensity_kg_per_kwh: float = DEFAULT_GRID_CARBON_INTENSITY_KG_PER_KWH
    source: str = "static_default"

    def to_carbon(self, energy_kwh: float) -> float:
        return energy_kwh * self.intensity_kg_per_kwh
