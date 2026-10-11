
import logging
import math

import requests


# ============================================================
# NEXORA — ROUTE INTELLIGENCE
# ============================================================

OSRM_URL = "https://router.project-osrm.org/route/v1/driving"

HEADERS = {
    "User-Agent": "NEXORA/1.0 (Site Intelligence Platform)",
    "Accept": "application/json",
}


# ============================================================
# COORDINATE VALIDATION
# ============================================================

def _validate_coordinates(
    latitude: float,
    longitude: float,
    label: str,
) -> tuple[float, float]:
    """Validate latitude and longitude."""

    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{label} coordinates must be numeric."
        ) from exc

    if not math.isfinite(latitude) or not -90 <= latitude <= 90:
        raise ValueError(
            f"{label} latitude must be between -90 and 90."
        )

    if not math.isfinite(longitude) or not -180 <= longitude <= 180:
        raise ValueError(
            f"{label} longitude must be between -180 and 180."
        )

    return latitude, longitude


# ============================================================
# GET ROUTE
# ============================================================

def get_route(
    origin_latitude: float,
    origin_longitude: float,
    destination_latitude: float,
    destination_longitude: float,
) -> dict:
    """Calculate a driving route using the OSRM routing service."""

    origin_latitude, origin_longitude = _validate_coordinates(
        origin_latitude,
        origin_longitude,
        "Origin",
    )

    destination_latitude, destination_longitude = _validate_coordinates(
        destination_latitude,
        destination_longitude,
        "Destination",
    )

    # OSRM expects longitude,latitude order.
    url = (
        f"{OSRM_URL}/"
        f"{origin_longitude},{origin_latitude};"
        f"{destination_longitude},{destination_latitude}"
    )

    params = {
        "overview": "full",
        "geometries": "geojson",
        "steps": "true",
    }

    try:
        logging.info("Requesting NEXORA route information")

        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=30,
        )
        response.raise_for_status()

    except requests.RequestException as exc:
        logging.error("OSRM request failed: %s", exc)
        raise RuntimeError(
            "Routing service is currently unavailable."
        ) from exc

    try:
        data = response.json()
    except (ValueError, requests.exceptions.JSONDecodeError) as exc:
        logging.error("OSRM returned invalid JSON.")
        raise RuntimeError(
            "Routing service returned an invalid response."
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            "Routing service returned an unexpected response."
        )

    if data.get("code") != "Ok":
        message = data.get(
            "message",
            "No valid route was returned.",
        )
        raise RuntimeError(
            f"Route calculation failed: {message}"
        )

    routes = data.get("routes")

    if not isinstance(routes, list) or not routes:
        raise RuntimeError(
            "No driving route was found between these locations."
        )

    route = routes[0]

    try:
        distance_m = float(route["distance"])
        duration_s = float(route["duration"])
        geometry = route["geometry"]
        legs = route.get("legs", [])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "Routing service returned incomplete route data."
        ) from exc

    if (
        not math.isfinite(distance_m)
        or not math.isfinite(duration_s)
        or distance_m < 0
        or duration_s < 0
    ):
        raise RuntimeError(
            "Routing service returned invalid distance or duration."
        )

    if (
        not isinstance(geometry, dict)
        or geometry.get("type") != "LineString"
        or not isinstance(geometry.get("coordinates"), list)
    ):
        raise RuntimeError(
            "Route geometry is missing or invalid."
        )

    if not isinstance(legs, list):
        legs = []

    return {
        "status": "success",
        "origin": {
            "latitude": origin_latitude,
            "longitude": origin_longitude,
        },
        "destination": {
            "latitude": destination_latitude,
            "longitude": destination_longitude,
        },
        "distance_km": round(distance_m / 1000, 3),
        "travel_time_min": round(duration_s / 60, 2),
        "geometry": geometry,
        "legs": legs,
        "source": "OSRM / OpenStreetMap",
        "travel_mode": "driving",
        "limitations": [
            "Travel time is an estimate from the routing engine.",
            "Live traffic conditions may not be reflected.",
            "Public OSRM service availability is not guaranteed.",
        ],
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        result = get_route(
            origin_latitude=18.5204,
            origin_longitude=73.8567,
            destination_latitude=19.0760,
            destination_longitude=72.8777,
        )

        print("\nNEXORA Route Analysis")
        print("---------------------")
        print(f"Status: {result['status']}")
        print(f"Distance: {result['distance_km']} km")
        print(f"Travel time: {result['travel_time_min']} minutes")
        print(f"Source: {result['source']}")

    except (ValueError, RuntimeError) as exc:
        print(f"\nRoute analysis failed: {exc}")
