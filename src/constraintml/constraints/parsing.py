"""Parsing helpers that turn user-facing constraint strings/numbers into typed values.

Every parser accepts either a bare number (assumed to already be in the target unit)
or a string like "8h", "<50kWh", "25 kg CO2e". A leading comparator ("<", "<=") is
informational only -- v1 treats every parsed budget as an upper bound regardless of
the comparator given. A leading ">"/">=" logs a warning since it is nonsensical for
a resource budget, but the numeric value is still used as an upper bound.
"""

from __future__ import annotations

import logging
import re
from numbers import Number

from .units import CARBON_UNITS_TO_KG, DURATION_UNITS_TO_SECONDS, ENERGY_UNITS_TO_KWH, normalize_unit

logger = logging.getLogger(__name__)

_VALUE_PATTERN = re.compile(r"^\s*([<>]=?)?\s*([\d.]+)\s*(.*?)\s*$")


def _split_value_unit(value: str | Number) -> tuple[float, str]:
    if isinstance(value, Number):
        return float(value), ""

    match = _VALUE_PATTERN.match(str(value))
    if not match:
        raise ValueError(f"Could not parse constraint value: {value!r}")

    comparator, number, unit = match.groups()
    if comparator in (">", ">="):
        logger.warning(
            "Constraint value %r uses comparator %r, which is nonsensical for an upper-bound "
            "budget; the numeric value will still be treated as an upper bound.",
            value,
            comparator,
        )

    number = float(number)
    if number < 0:
        raise ValueError(f"Constraint value must be non-negative, got: {value!r}")

    return number, unit


def parse_duration(value: str | Number) -> float:
    """Parse a duration like "8h", "30m", "1d", or a bare number of seconds -> seconds."""
    number, unit = _split_value_unit(value)
    unit = normalize_unit(unit)
    if unit == "":
        return number
    if unit not in DURATION_UNITS_TO_SECONDS:
        raise ValueError(f"Unknown duration unit {unit!r} in {value!r}")
    return number * DURATION_UNITS_TO_SECONDS[unit]


def parse_energy(value: str | Number) -> float:
    """Parse an energy budget like "100kWh", "<50kWh", or a bare number -> kWh."""
    number, unit = _split_value_unit(value)
    unit = normalize_unit(unit)
    if unit not in ENERGY_UNITS_TO_KWH:
        raise ValueError(f"Unknown energy unit {unit!r} in {value!r}")
    return number * ENERGY_UNITS_TO_KWH[unit]


def parse_carbon(value: str | Number) -> float:
    """Parse a carbon budget like "25 kg CO2e", "8kgCO2e", or a bare number -> kg CO2e."""
    number, unit = _split_value_unit(value)
    unit = normalize_unit(unit)
    if unit not in CARBON_UNITS_TO_KG:
        raise ValueError(f"Unknown carbon unit {unit!r} in {value!r}")
    return number * CARBON_UNITS_TO_KG[unit]


def parse_cost(value: str | Number) -> float:
    """Parse a cost budget like "$50", "50 USD", or a bare number -> USD."""
    if isinstance(value, Number):
        number = float(value)
    else:
        cleaned = str(value).strip().lstrip("$").replace(",", "")
        cleaned = re.sub(r"(?i)\s*usd\s*$", "", cleaned)
        number, _unit = _split_value_unit(cleaned)
    if number < 0:
        raise ValueError(f"Cost budget must be non-negative, got: {value!r}")
    return number


def parse_percentage(value: str | Number) -> float:
    """Parse a percentage like "0.5%" -> 0.005. A bare number < 1 is assumed to already be
    a fraction; a bare number >= 1 is assumed to be a percentage (e.g. 5 -> 0.05)."""
    number, unit = _split_value_unit(value)
    unit = normalize_unit(unit)
    if unit == "%":
        return number / 100
    if unit != "":
        raise ValueError(f"Unknown percentage unit {unit!r} in {value!r}")
    return number if number < 1 else number / 100
