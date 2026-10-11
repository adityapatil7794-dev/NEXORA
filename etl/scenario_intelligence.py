
from __future__ import annotations

from typing import Any


SUPPORTED_SCENARIOS = {
    "baseline": {
        "label": "Baseline",
        "description": "Current observed conditions and available evidence.",
    },
    "increased_rainfall": {
        "label": "Increased Rainfall",
        "description": (
            "Sensitivity scenario for increased rainfall. "
            "This is not a location-specific climate forecast."
        ),
        "rainfall_multiplier": 1.20,
    },
    "extreme_rainfall": {
        "label": "Extreme Rainfall",
        "description": (
            "Stress-test scenario for substantially increased rainfall. "
            "This is not a probability forecast."
        ),
        "rainfall_multiplier": 1.50,
    },
}


def analyze_scenarios(
    *,
    climate: dict[str, Any],
    flood: dict[str, Any],
) -> dict[str, Any]:
    """Compare rainfall sensitivity scenarios without inventing predictions."""

    rainfall = climate.get("climate", {}).get("precipitation_sum")

    if rainfall is None:
        return {
            "status": "DATA LIMITED",
            "scenarios": [],
            "limitations": [
                "Observed precipitation data is unavailable.",
                "Scenario comparisons cannot be calculated.",
            ],
        }

    try:
        rainfall = float(rainfall)
    except (TypeError, ValueError):
        return {
            "status": "DATA LIMITED",
            "scenarios": [],
            "limitations": ["Precipitation data is invalid."],
        }

    if rainfall < 0:
        return {
            "status": "DATA LIMITED",
            "scenarios": [],
            "limitations": ["Precipitation cannot be negative."],
        }

    results = [
        {
            "scenario_id": "baseline",
            "label": "Baseline",
            "observed_precipitation_mm": round(rainfall, 2),
            "scenario_precipitation_mm": round(rainfall, 2),
            "change_percent": 0,
            "interpretation": (
                "Reference value based on the supplied climate observations."
            ),
        }
    ]

    for scenario_id in ("increased_rainfall", "extreme_rainfall"):
        scenario = SUPPORTED_SCENARIOS[scenario_id]
        multiplier = scenario["rainfall_multiplier"]
        projected = rainfall * multiplier

        results.append(
            {
                "scenario_id": scenario_id,
                "label": scenario["label"],
                "observed_precipitation_mm": round(rainfall, 2),
                "scenario_precipitation_mm": round(projected, 2),
                "change_percent": round((multiplier - 1) * 100, 1),
                "interpretation": scenario["description"],
            }
        )

    return {
        "status": "success",
        "scenario_type": "rainfall_sensitivity",
        "scenarios": results,
        "flood_evidence_status": flood.get("risk_status", "UNKNOWN"),
        "limitations": [
            "Rainfall multipliers are hypothetical stress-test assumptions.",
            "They are not climate projections or flood probabilities.",
            "Flood probability must not be recalculated from rainfall alone.",
        ],
        "source": "NEXORA scenario sensitivity analysis",
    }
