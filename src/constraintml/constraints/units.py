"""Unit lookup tables used by constraints.parsing."""

DURATION_UNITS_TO_SECONDS = {
    "s": 1,
    "sec": 1,
    "secs": 1,
    "second": 1,
    "seconds": 1,
    "m": 60,
    "min": 60,
    "mins": 60,
    "minute": 60,
    "minutes": 60,
    "h": 3600,
    "hr": 3600,
    "hrs": 3600,
    "hour": 3600,
    "hours": 3600,
    "d": 86400,
    "day": 86400,
    "days": 86400,
}

# All keys normalized (lowercase, spaces/dashes/underscores stripped) to kWh.
ENERGY_UNITS_TO_KWH = {
    "kwh": 1.0,
    "wh": 1e-3,
    "mwh": 1e3,
    "j": 1 / 3_600_000,
    "joule": 1 / 3_600_000,
    "joules": 1 / 3_600_000,
    "kj": 1e3 / 3_600_000,
    "mj": 1e6 / 3_600_000,
    "": 1.0,  # bare number defaults to kWh
}

# All keys normalized (lowercase, spaces/dashes/underscores stripped) to kg CO2e.
CARBON_UNITS_TO_KG = {
    "kgco2e": 1.0,
    "kgco2": 1.0,
    "kg": 1.0,
    "gco2e": 1e-3,
    "gco2": 1e-3,
    "g": 1e-3,
    "tco2e": 1e3,
    "tco2": 1e3,
    "t": 1e3,
    "": 1.0,  # bare number defaults to kg CO2e
}


def normalize_unit(unit: str) -> str:
    """Lowercase and strip spaces/dashes/underscores so 'kg CO2e' == 'kgCO2e' == 'kg-co2e'."""
    return unit.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
