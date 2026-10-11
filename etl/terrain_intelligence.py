from __future__ import annotations

import math
from typing import Any

import requests


OPEN_METEO_ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"


def validate_coordinates(latitude: float, longitude: float) -> None:
    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")

    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180.")


def _elevation_points(
    latitude: float,
    longitude: float,
    offset: float = 0.002,
) -> list[tuple[float, float]]:
    """
    Create a small cross around the selected location.

    Center + north + south + east + west.
    """

    return [
        (latitude, longitude),
        (latitude + offset, longitude),
        (latitude - offset, longitude),
        (latitude, longitude + offset),
        (latitude, longitude - offset),
    ]


def _fetch_elevations(
    points: list[tuple[float, float]],
) -> list[float]:

    latitude = ",".join(str(point[0]) for point in points)
    longitude = ",".join(str(point[1]) for point in points)

    response = requests.get(
        OPEN_METEO_ELEVATION_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
        },
        timeout=15,
    )

    response.raise_for_status()

    payload = response.json()

    elevations = payload.get("elevation")

    if not elevations:
        raise ValueError("Elevation service returned no elevation data.")

    return [float(value) for value in elevations]


def _distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:

    radius = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )

    return radius * 2 * math.asin(math.sqrt(a))


def _classify_slope(slope_percent: float) -> str:

    if slope_percent < 2:
        return "VERY FLAT"

    if slope_percent < 5:
        return "GENTLE"

    if slope_percent < 10:
        return "MODERATE"

    if slope_percent < 20:
        return "STEEP"

    return "VERY STEEP"


def analyze_terrain(
    *,
    latitude: float,
    longitude: float,
) -> dict[str, Any]:

    validate_coordinates(latitude, longitude)

    points = _elevation_points(
        latitude,
        longitude,
    )

    elevations = _fetch_elevations(points)

    center = elevations[0]
    north = elevations[1]
    south = elevations[2]
    east = elevations[3]
    west = elevations[4]

    elevation_min = min(elevations)
    elevation_max = max(elevations)

    elevation_range = elevation_max - elevation_min

    north_south_distance = _distance_km(
        latitude - 0.002,
        longitude,
        latitude + 0.002,
        longitude,
    )

    east_west_distance = _distance_km(
        latitude,
        longitude - 0.002,
        latitude,
        longitude + 0.002,
    )

    ns_slope = (
        abs(north - south)
        / (north_south_distance * 1000)
    ) * 100

    ew_slope = (
        abs(east - west)
        / (east_west_distance * 1000)
    ) * 100

    slope_percent = max(
        ns_slope,
        ew_slope,
    )

    terrain_class = _classify_slope(
        slope_percent
    )

    return {
        "status": "success",

        "location": {
            "latitude": latitude,
            "longitude": longitude,
        },

        "elevation": {
            "center_m": round(center, 2),
            "minimum_m": round(elevation_min, 2),
            "maximum_m": round(elevation_max, 2),
            "range_m": round(elevation_range, 2),
        },

        "slope": {
            "estimated_percent": round(
                slope_percent,
                2,
            ),
            "terrain_class": terrain_class,
        },

        "sample_points": {
            "north_m": round(north, 2),
            "south_m": round(south, 2),
            "east_m": round(east, 2),
            "west_m": round(west, 2),
        },

        "source": "Open-Meteo Elevation API",

        "confidence": "MODERATE",

        "interpretation": (
            "Terrain values are remotely derived estimates "
            "and should not replace detailed topographic or "
            "engineering surveys."
        ),
    }