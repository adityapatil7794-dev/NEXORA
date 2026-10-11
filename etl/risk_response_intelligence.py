from __future__ import annotations

from typing import Any


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        if result != result or result in (float("inf"), float("-inf")):
            return None
        return result
    except (TypeError, ValueError):
        return None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def build_risk_response(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Convert NEXORA evidence into risk signals and due-diligence options.

    This function does not certify site safety or invent measurements.
    """

    site = _mapping(payload.get("site_intelligence", payload))

    climate_response = _mapping(site.get("climate"))
    climate = _mapping(climate_response.get("climate"))
    climate = climate or climate_response

    flood = _mapping(site.get("historical_flood"))
    flood_ml = _mapping(site.get("flood_ml"))
    prediction = _mapping(
        flood_ml.get("prediction") or flood_ml.get("ml_prediction")
    )

    if not prediction:
        prediction = flood_ml

    soil = _mapping(site.get("soil"))
    terrain = _mapping(payload.get("terrain") or site.get("terrain"))
    infrastructure = _mapping(
        payload.get("infrastructure") or site.get("infrastructure")
    )
    scenarios = _mapping(
        payload.get("scenarios") or site.get("scenarios")
    )

    risks: list[dict[str, Any]] = []
    investigations: list[dict[str, Any]] = []

    def add_risk(
        risk_id: str,
        title: str,
        level: str,
        evidence: list[dict[str, Any]],
        response_options: list[str],
    ) -> None:
        risks.append(
            {
                "risk_id": risk_id,
                "title": title,
                "level": level,
                "evidence": evidence,
                "response_options": response_options,
                "requires_professional_review": True,
            }
        )

    def add_investigation(
        investigation_id: str,
        title: str,
        reason: str,
        priority: str = "RECOMMENDED",
    ) -> None:
        investigations.append(
            {
                "investigation_id": investigation_id,
                "title": title,
                "reason": reason,
                "priority": priority,
            }
        )

    # 1. Flood model evidence
    probability_percent = None

    for key in ("probability_percent", "flood_probability_percent"):
        probability_percent = _number(prediction.get(key))
        if probability_percent is not None:
            break

    if probability_percent is None:
        for key in ("probability", "flood_probability"):
            probability = _number(prediction.get(key))
            if probability is not None:
                probability_percent = (
                    probability * 100 if 0 <= probability <= 1
                    else probability
                )
                break

    if probability_percent is not None:
        if not 0 <= probability_percent <= 100:
            probability_percent = None

    if probability_percent is not None:
        if probability_percent >= 80:
            level = "VERY HIGH"
        elif probability_percent >= 60:
            level = "HIGH"
        elif probability_percent >= 40:
            level = "ELEVATED"
        elif probability_percent >= 20:
            level = "MODERATE"
        else:
            level = "LOW"

        add_risk(
            "FLOOD_ML_SIGNAL",
            "Flood model signal",
            level,
            [
                {
                    "type": "model_probability",
                    "value_percent": round(probability_percent, 2),
                    "source": flood_ml.get(
                        "source", "NEXORA flood model response"
                    ),
                }
            ],
            [
                "Arrange a site-specific hydrological and flood assessment.",
                "Verify official flood maps, drainage paths, and local flood records.",
                "Have a qualified engineer evaluate suitable drainage and flood mitigation options.",
            ],
        )
        add_investigation(
            "FLOOD_ASSESSMENT",
            "Site-specific flood assessment",
            "A model signal is not a certified flood-depth or flood-safety assessment.",
            "HIGH" if probability_percent >= 60 else "RECOMMENDED",
        )
    else:
        add_investigation(
            "FLOOD_MODEL_DATA",
            "Verify flood model evidence",
            "A usable flood-model probability was not supplied.",
        )

    # 2. Historical flood evidence
    event_count = _number(flood.get("event_count"))
    historical_status = str(flood.get("risk_status", "UNKNOWN")).upper()

    if (event_count is not None and event_count > 0) or (
        historical_status == "EVENTS DETECTED"
    ):
        add_risk(
            "HISTORICAL_FLOOD_EVIDENCE",
            "Mapped historical flood events",
            "ELEVATED",
            [
                {
                    "type": "historical_events",
                    "event_count": int(event_count)
                    if event_count is not None else None,
                    "radius_km": flood.get("radius_km"),
                    "source": _mapping(flood.get("dataset")).get(
                        "name", "Historical flood response"
                    ),
                }
            ],
            [
                "Review the mapped event locations, dates, and distances.",
                "Verify local flood records and site drainage with qualified professionals.",
            ],
        )
        add_investigation(
            "HISTORICAL_FLOOD_REVIEW",
            "Review historical flood evidence",
            "Mapped events require site-specific interpretation.",
            "HIGH",
        )
    elif historical_status in ("DATA LIMITED", "UNKNOWN"):
        add_investigation(
            "HISTORICAL_FLOOD_DATA",
            "Obtain reliable historical flood evidence",
            "Historical flood coverage is limited or its status is unknown.",
        )

    # 3. Heavy-rain indicators
    heavy_days = _number(climate.get("heavy_rain_days"))
    extreme_days = _number(climate.get("extreme_rain_days"))

    if (
        (heavy_days is not None and heavy_days > 0)
        or (extreme_days is not None and extreme_days > 0)
    ):
        add_risk(
            "HEAVY_RAIN_SIGNAL",
            "Heavy-rain observations",
            "REVIEW",
            [
                {
                    "heavy_rain_days": heavy_days,
                    "extreme_rain_days": extreme_days,
                    "precipitation_max": climate.get("precipitation_max"),
                    "source": climate_response.get(
                        "source", "Climate response"
                    ),
                }
            ],
            [
                "Have a drainage specialist assess stormwater capacity.",
                "Check local rainfall design standards and surface-water flow paths.",
            ],
        )
        add_investigation(
            "STORMWATER_REVIEW",
            "Stormwater and drainage review",
            "Observed heavy-rain indicators warrant local drainage assessment.",
        )

    # 4. Terrain and slope
    slope = _mapping(terrain.get("slope") or terrain.get("Slope"))
    slope_percent = _number(slope.get("estimated_percent"))
    terrain_class = str(slope.get("terrain_class", "")).upper()

    if (
        (slope_percent is not None and slope_percent >= 15)
        or terrain_class in ("STEEP", "VERY STEEP")
    ):
        add_risk(
            "TERRAIN_SLOPE",
            "Terrain slope requires review",
            "REVIEW",
            [
                {
                    "slope_percent": slope_percent,
                    "terrain_class": terrain_class or None,
                    "source": terrain.get("source", "Terrain response"),
                }
            ],
            [
                "Commission a detailed topographic and geotechnical survey.",
                "Obtain a site-specific slope-stability assessment where relevant.",
            ],
        )
        add_investigation(
            "SLOPE_STABILITY",
            "Terrain and slope-stability assessment",
            "Remote slope estimates do not replace a detailed site survey.",
        )

    if not terrain:
        add_investigation(
            "TERRAIN_DATA",
            "Obtain terrain evidence",
            "Terrain analysis was not supplied to the risk-response module.",
        )

    # 5. Soil and geotechnical evidence
    bearing_capacity = _number(soil.get("bearing_capacity"))
    soil_confidence = str(
        soil.get("data_confidence", "")
    ).upper()

    if not soil or bearing_capacity is None:
        add_risk(
            "GEOTECHNICAL_DATA_GAP",
            "Geotechnical evidence is incomplete",
            "UNKNOWN",
            [
                {
                    "bearing_capacity_available": bearing_capacity is not None,
                    "soil_record_available": bool(soil),
                    "data_confidence": soil_confidence or None,
                }
            ],
            [
                "Obtain a qualified geotechnical investigation.",
                "Verify bearing capacity and other engineering parameters through appropriate field or laboratory testing.",
            ],
        )
        add_investigation(
            "GEOTECHNICAL_INVESTIGATION",
            "Obtain verified soil and geotechnical data",
            "GPS and remote soil datasets cannot certify foundation suitability.",
            "HIGH",
        )
    elif soil_confidence == "LOW":

        add_investigation(
            "GEOTECHNICAL_VERIFICATION",
            "Verify low-confidence soil evidence.",
            "The available soil record has low stated confidence.",
        )

    # 6. Infrastructure evidence
    if not infrastructure:
        add_investigation(
            "INFRASTRUCTURE_VERIFICATION",
            "Verify infrastructure and utilities",
            "Infrastructure analysis was not supplied to this module.",
        )
    else:
        add_investigation(
            "INFRASTRUCTURE_DUE_DILIGENCE",
            "Verify project-level infrastructure access",
            "Mapped proximity does not confirm capacity, legal access, or service availability.",
        )

    # 7. Future scenario evidence
    if scenarios:
        add_investigation(
            "SCENARIO_REVIEW",
            "Review future sensitivity scenarios",
            "Confirm scenario assumptions and compare them with authoritative climate projections.",
        )

    # Determine available evidence
    module_data = {
        "CLIMATE": climate_response,
        "HISTORICAL_FLOOD": flood,
        "FLOOD_ML": flood_ml,
        "SOIL": soil,
        "TERRAIN": terrain,
        "INFRASTRUCTURE": infrastructure,
        "SCENARIOS": scenarios,
    }

    available_modules = [
        name for name, data in module_data.items() if data
    ]

    missing_modules = [
        name for name, data in module_data.items() if not data
    ]

    if not available_modules:
        status = "DATA LIMITED"
    elif risks or investigations:
        status = "REVIEW REQUIRED"
    else:
        status = "NO MATERIAL SIGNALS DETECTED"

    confidence = (
        "MODERATE"
        if len(available_modules) >= 5 and not missing_modules
        else "LOW"
    )

    return {
        "status": status,
        "confidence": confidence,
        "risks": risks,
        "recommended_investigations": investigations,
        "available_modules": available_modules,
        "missing_modules": missing_modules,
        "interpretation": (
            "These are evidence-linked risk signals and due-diligence options, "
            "not engineering instructions or a site safety certification. "
            "Missing evidence reduces confidence."
        ),
    }