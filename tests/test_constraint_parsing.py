import pytest

from constraintml.constraints.parsing import (
    parse_carbon,
    parse_cost,
    parse_duration,
    parse_energy,
    parse_percentage,
)
from constraintml.constraints.spec import ConstraintSpec


@pytest.mark.parametrize(
    "value,expected_seconds",
    [
        ("8h", 8 * 3600),
        ("30m", 30 * 60),
        ("1d", 86400),
        (120, 120),
        ("120", 120),
    ],
)
def test_parse_duration(value, expected_seconds):
    assert parse_duration(value) == pytest.approx(expected_seconds)


@pytest.mark.parametrize(
    "value,expected_kwh",
    [
        ("100kWh", 100),
        ("<50kWh", 50),
        (100, 100),
        ("100", 100),
        ("1000Wh", 1),
    ],
)
def test_parse_energy(value, expected_kwh):
    assert parse_energy(value) == pytest.approx(expected_kwh)


@pytest.mark.parametrize(
    "value,expected_kg",
    [
        ("25", 25),
        ("25 kg CO2e", 25),
        ("8kgCO2e", 8),
        ("8000 g CO2e", 8),
        ("8kg-co2e", 8),
    ],
)
def test_parse_carbon(value, expected_kg):
    assert parse_carbon(value) == pytest.approx(expected_kg)


@pytest.mark.parametrize(
    "value,expected_fraction",
    [
        ("0.5%", 0.005),
        ("5%", 0.05),
        (0.005, 0.005),
        (5, 0.05),
    ],
)
def test_parse_percentage(value, expected_fraction):
    assert parse_percentage(value) == pytest.approx(expected_fraction)


def test_parse_cost():
    assert parse_cost("$50") == pytest.approx(50)
    assert parse_cost("50 USD") == pytest.approx(50)
    assert parse_cost(50) == pytest.approx(50)


@pytest.mark.parametrize("value", ["100 furlongs", "abc", "10 lightyears"])
def test_bad_unit_raises(value):
    with pytest.raises(ValueError):
        parse_energy(value)


def test_negative_value_raises():
    with pytest.raises(ValueError):
        parse_energy("-5kWh")


def test_constraint_spec_from_dict_matches_readme_example():
    spec = ConstraintSpec.from_dict(
        {
            "energy": "<50kWh",
            "carbon": "<10kgCO2e",
            "deadline": "8h",
            "max_accuracy_loss": "0.5%",
        }
    )
    assert spec.energy_budget_kwh == pytest.approx(50)
    assert spec.carbon_budget_kgco2e == pytest.approx(10)
    assert spec.deadline_seconds == pytest.approx(8 * 3600)
    assert spec.max_accuracy_loss == pytest.approx(0.005)


def test_constraint_spec_kwargs_override_dict():
    dict_spec = ConstraintSpec.from_dict({"energy": "50kWh"})
    kwargs_spec = ConstraintSpec.from_kwargs(energy_budget="100kWh")
    merged = dict_spec.merge(kwargs_spec)
    assert merged.energy_budget_kwh == pytest.approx(100)


def test_constraint_spec_unknown_key_raises():
    with pytest.raises(ValueError):
        ConstraintSpec.from_dict({"bogus": "1"})
