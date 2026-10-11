from __future__ import annotations

import math
from typing import Any

import requests


# ============================================================
# NEXORA — INFRASTRUCTURE INTELLIGENCE
# ============================================================

OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

HEADERS = {
    "User-Agent": "NEXORA/1.0 (site intelligence platform)",
    "Accept": "application/json",
    "Content-Type": "application/x-www-form-urlencoded",
}

CATEGORIES = (
    "motorways",
    "railways",
    "railway_stations",
    "airports",
    "hospitals",
    "power_substations",
)


# ============================================================
# DISTANCE CALCULATION
# ============================================================

def _distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate straight-line distance in kilometres."""

    earth_radius_km = 6371.0088

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(delta_lon / 2) ** 2
    )

    a = min(1.0, max(0.0, a))

    return (
        2
        * earth_radius_km
        * math.asin(math.sqrt(a))
    )


# ============================================================
# OVERPASS API REQUEST WITH FAILOVER
# ============================================================

def _query_overpass(
    query: str,
) -> list[dict[str, Any]]:
    """Query Overpass API using alternative public endpoints."""

    last_error = None

    for endpoint in OVERPASS_ENDPOINTS:
        try:
            response = requests.post(
                endpoint,
                data={"data": query},
                headers=HEADERS,
                timeout=30,
            )

            if response.status_code != 200:
                last_error = (
                    f"{endpoint} returned HTTP "
                    f"{response.status_code}"
                )
                continue

            payload = response.json()

            elements = payload.get("elements", [])

            if isinstance(elements, list):
                return [
                    item
                    for item in elements
                    if isinstance(item, dict)
                ]

            last_error = (
                f"{endpoint} returned invalid elements data"
            )

        except requests.RequestException as exc:
            last_error = f"{endpoint}: {exc}"

        except (ValueError, TypeError) as exc:
            last_error = (
                f"{endpoint}: invalid JSON response: {exc}"
            )

    print(
        "NEXORA Overpass warning: "
        f"{last_error or 'all providers failed'}"
    )

    return []


# ============================================================
# NORMALIZE OPENSTREETMAP ELEMENT
# ============================================================

def _normalize_element(
    element: dict[str, Any],
    latitude: float,
    longitude: float,
) -> dict[str, Any] | None:
    """Normalize an OSM element and calculate its distance."""

    tags = element.get("tags") or {}

    if not isinstance(tags, dict):
        tags = {}

    lat = element.get("lat")
    lon = element.get("lon")

    # Ways and relations may provide coordinates through center.
    if lat is None or lon is None:
        center = element.get("center") or {}

        if isinstance(center, dict):
            lat = center.get("lat")
            lon = center.get("lon")

    if lat is None or lon is None:
        return None

    try:
        asset_lat = float(lat)
        asset_lon = float(lon)

    except (TypeError, ValueError):
        return None

    if not -90 <= asset_lat <= 90:
        return None

    if not -180 <= asset_lon <= 180:
        return None

    distance = _distance_km(
        latitude,
        longitude,
        asset_lat,
        asset_lon,
    )

    return {
        "name": tags.get("name"),
        "type": element.get("type"),
        "osm_id": element.get("id"),
        "latitude": asset_lat,
        "longitude": asset_lon,
        "distance_km": round(distance, 3),
    }


# ============================================================
# MAIN INFRASTRUCTURE ANALYSIS
# ============================================================

def analyze_infrastructure(
    latitude: float,
    longitude: float,
    radius_km: float = 25.0,
) -> dict[str, Any]:
    """
    Analyse infrastructure around a selected project location.

    All six infrastructure categories use one Overpass query.
    The request helper handles endpoint failover.
    """

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    try:
        latitude = float(latitude)
        longitude = float(longitude)
        radius_km = float(radius_km)

    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Latitude, longitude and radius must be numeric."
        ) from exc

    if not math.isfinite(latitude):
        raise ValueError("Latitude must be finite.")

    if not -90 <= latitude <= 90:
        raise ValueError(
            "Latitude must be between -90 and 90."
        )

    if not math.isfinite(longitude):
        raise ValueError("Longitude must be finite.")

    if not -180 <= longitude <= 180:
        raise ValueError(
            "Longitude must be between -180 and 180."
        )

    if not math.isfinite(radius_km) or radius_km <= 0:
        raise ValueError(
            "Radius must be greater than zero."
        )

    # Limit request size for public Overpass services.
    radius_km = min(radius_km, 50.0)
    radius_m = int(radius_km * 1000)

    # --------------------------------------------------------
    # INITIALIZE CATEGORIES
    # --------------------------------------------------------

    infrastructure: dict[
        str, list[dict[str, Any]]
    ] = {
        category: []
        for category in CATEGORIES
    }

    # --------------------------------------------------------
    # SINGLE COMBINED OVERPASS QUERY
    # --------------------------------------------------------

    query = f"""
    [out:json][timeout:30];
    (
      node(around:{radius_m},{latitude},{longitude})
        ["highway"="motorway"];
      way(around:{radius_m},{latitude},{longitude})
        ["highway"="motorway"];

      node(around:{radius_m},{latitude},{longitude})
        ["railway"="rail"];
      way(around:{radius_m},{latitude},{longitude})
        ["railway"="rail"];

      node(around:{radius_m},{latitude},{longitude})
        ["railway"="station"];
      way(around:{radius_m},{latitude},{longitude})
        ["railway"="station"];

      node(around:{radius_m},{latitude},{longitude})
        ["aeroway"="aerodrome"];
      way(around:{radius_m},{latitude},{longitude})
        ["aeroway"="aerodrome"];

      node(around:{radius_m},{latitude},{longitude})
        ["amenity"="hospital"];
      way(around:{radius_m},{latitude},{longitude})
        ["amenity"="hospital"];

      node(around:{radius_m},{latitude},{longitude})
        ["power"="substation"];
      way(around:{radius_m},{latitude},{longitude})
        ["power"="substation"];
    );
    out center tags;
    """

    elements = _query_overpass(query)

    # --------------------------------------------------------
    # CATEGORY CLASSIFICATION
    # --------------------------------------------------------

    category_rules = (
        (
            "motorways",
            lambda tags: tags.get("highway") == "motorway",
        ),
        (
            "railways",
            lambda tags: tags.get("railway") == "rail",
        ),
        (
            "railway_stations",
            lambda tags: tags.get("railway") == "station",
        ),
        (
            "airports",
            lambda tags: tags.get("aeroway") == "aerodrome",
        ),
        (
            "hospitals",
            lambda tags: tags.get("amenity") == "hospital",
        ),
        (
            "power_substations",
            lambda tags: tags.get("power") == "substation",
        ),
    )

    # --------------------------------------------------------
    # NORMALIZE AND CLASSIFY RESULTS
    # --------------------------------------------------------

    seen: set[tuple[str, Any]] = set()

    for element in elements:
        tags = element.get("tags") or {}

        if not isinstance(tags, dict):
            continue

        item = _normalize_element(
            element,
            latitude,
            longitude,
        )

        if item is None:
            continue

        identity = (
            str(element.get("type", "")),
            element.get("id"),
        )

        if identity in seen:
            continue

        seen.add(identity)

        for category, matches in category_rules:
            if matches(tags):
                infrastructure[category].append(item)
                break

    # --------------------------------------------------------
    # SORT BY DISTANCE
    # --------------------------------------------------------

    for items in infrastructure.values():
        items.sort(
            key=lambda item: item["distance_km"]
        )

    # --------------------------------------------------------
    # COUNTS
    # --------------------------------------------------------

    counts = {
        category: len(items)
        for category, items in infrastructure.items()
    }

    total_assets = sum(counts.values())

    # --------------------------------------------------------
    # NEAREST ASSET PER CATEGORY
    # --------------------------------------------------------

    nearest = {
        category: items[0] if items else None
        for category, items in infrastructure.items()
    }

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "status": "success",
        "location": {
            "latitude": latitude,
            "longitude": longitude,
        },
        "radius_km": radius_km,
        "infrastructure": infrastructure,
        "counts": counts,
        "total_infrastructure_assets": total_assets,
        "nearest": nearest,
        "source": "OpenStreetMap / Overpass API",
        "confidence": "MODERATE",
        "data_limitations": [
            "OpenStreetMap coverage varies by location.",
            (
                "Distances are straight-line geographic "
                "distances, not road travel distances."
            ),
            (
                "Project-level engineering and utility "
                "verification is still required."
            ),
        ],
        "interpretation": (
            "Infrastructure intelligence provides mapped "
            "proximity evidence for roads, railways, stations, "
            "airports, hospitals and power substations. "
            "Verify findings against authoritative "
            "project-level data."
        ),
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":
    import json

    result = analyze_infrastructure(
        latitude=16.723,
        longitude=74.543,
        radius_km=25,
    )

    print(json.dumps(result, indent=2))