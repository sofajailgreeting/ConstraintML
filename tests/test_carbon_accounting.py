import pytest

from constraintml.optimization.carbon import CarbonModel, DEFAULT_GRID_CARBON_INTENSITY_KG_PER_KWH


def test_default_intensity():
    model = CarbonModel()
    assert model.intensity_kg_per_kwh == DEFAULT_GRID_CARBON_INTENSITY_KG_PER_KWH


def test_to_carbon_arithmetic():
    model = CarbonModel(intensity_kg_per_kwh=0.5)
    assert model.to_carbon(10) == pytest.approx(5.0)


def test_to_carbon_override():
    model = CarbonModel(intensity_kg_per_kwh=0.1, source="custom_grid")
    assert model.to_carbon(20) == pytest.approx(2.0)
    assert model.source == "custom_grid"
