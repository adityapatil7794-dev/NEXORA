
"""Historical earthquake intelligence for NEXORA.

Data source: USGS Earthquake Catalog.
Returns structured event data with pagination and error handling.
"""

from datetime import datetime, timezone
import math

import requests


USGS_API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

SOURCE_NAME = "USGS Earthquake Catalog"
PAGE_SIZE = 2000
USGS_MAX_LIMIT = 20000
MAX_RADIUS_KM = 20001.6
REQUEST_TIMEOUT = 30


def _failure(message, radius_km, start_time, min_magnitude):
    """Return a consistent failure response."""
    return {
        "status": "DATA UNAVAILABLE",
        "source": SOURCE_NAME,
        "radius_km": radius_km,
        "start_time": start_time,
        "min_magnitude": min_magnitude,
        "count": 0,
        "events": [],
        "complete": False,
        "truncated": False,
        "error": message,
    }


def _validate_start_time(start_time):
    """Validate an ISO date or ISO datetime."""
    if not isinstance(start_time, str) or not start_time.strip():
        raise ValueError("start_time must be a non-empty ISO date or datetime.")

    value = start_time.strip()

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "start_time must use ISO format, e.g. 2000-01-01."
        ) from exc

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)

    if parsed > datetime.now(timezone.utc).replace(tzinfo=None):
        raise ValueError("start_time cannot be in the future.")

    return value


def get_historical_earthquakes(
    latitude: float,
    longitude: float,
    radius_km: float = 100,
    start_time: str = "2000-01-01",
    min_magnitude: float = 2.5,
    max_events: int = USGS_MAX_LIMIT,
):
    """Fetch historical earthquakes around a geographic point.

    Args:
        latitude: Site latitude in degrees.
        longitude: Site longitude in degrees.
        radius_km: Search radius in kilometers.
        start_time: ISO date or datetime.
        min_magnitude: Minimum earthquake magnitude.
        max_events: Maximum number of events to retrieve (up to 20,000).

    Returns:
        A dictionary containing status, source, events, count and metadata.

    Raises:
        ValueError: If an input parameter is invalid.
    """

    # Validate numeric inputs, including NaN and infinity.
    numeric_values = {
        "latitude": latitude,
        "longitude": longitude,
        "radius_km": radius_km,
        "min_magnitude": min_magnitude,
    }

    for name, value in numeric_values.items():
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a valid number.")

        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a valid number.") from exc

        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite.")

        numeric_values[name] = number

    latitude = numeric_values["latitude"]
    longitude = numeric_values["longitude"]
    radius_km = numeric_values["radius_km"]
    min_magnitude = numeric_values["min_magnitude"]

    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")

    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180.")

    if not 0 < radius_km <= MAX_RADIUS_KM:
        raise ValueError(
            f"radius_km must be greater than 0 and at most {MAX_RADIUS_KM}."
        )

    if not -2 <= min_magnitude <= 10:
        raise ValueError("min_magnitude must be between -2 and 10.")

    if isinstance(max_events, bool) or not isinstance(max_events, int):
        raise ValueError("max_events must be an integer.")

    if not 1 <= max_events <= USGS_MAX_LIMIT:
        raise ValueError(
            f"max_events must be between 1 and {USGS_MAX_LIMIT}."
        )

    start_time = _validate_start_time(start_time)

    events = []
    seen_ids = set()
    offset = 1
    truncated = False

    try:
        with requests.Session() as session:
            session.headers.update({
                "User-Agent": "NEXORA-Site-Intelligence/1.0"
            })

            while len(events) < max_events:
                page_limit = min(
                    PAGE_SIZE,
                    max_events - len(events),
                )

                params = {
                    "format": "geojson",
                    "latitude": latitude,
                    "longitude": longitude,
                    "maxradiuskm": radius_km,
                    "starttime": start_time,
                    "minmagnitude": min_magnitude,
                    "orderby": "time-asc",
                    "limit": page_limit,
                    "offset": offset,
                }

                response = session.get(
                    USGS_API_URL,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()

                if not isinstance(data, dict):
                    raise ValueError("USGS returned an invalid response.")

                features = data.get("features")

                if not isinstance(features, list):
                    raise ValueError(
                        "USGS response does not contain a valid features list."
                    )

                for feature in features:
                    if not isinstance(feature, dict):
                        continue

                    event_id = feature.get("id")

                    if not event_id or event_id in seen_ids:
                        continue

                    properties = feature.get("properties") or {}
                    geometry = feature.get("geometry") or {}
                    coordinates = geometry.get("coordinates") or []

                    if not isinstance(properties, dict):
                        continue

                    if not isinstance(coordinates, (list, tuple)):
                        coordinates = []

                    def coordinate(index):
                        if len(coordinates) <= index:
                            return None

                        value = coordinates[index]

                        if isinstance(value, bool):
                            return None

                        if not isinstance(value, (int, float)):
                            return None

                        return value if math.isfinite(value) else None

                    events.append({
                        "event_id": event_id,
                        "magnitude": properties.get("mag"),
                        "place": properties.get("place"),
                        "time_ms": properties.get("time"),
                        "longitude": coordinate(0),
                        "latitude": coordinate(1),
                        "depth_km": coordinate(2),
                        "url": properties.get("url"),
                    })

                    seen_ids.add(event_id)

                received = len(features)
                offset += received

                if received < page_limit:
                    break

                if len(events) >= max_events:
                    # We reached our configured cap. Do not claim
                    # that the full matching history was retrieved.
                    truncated = True
                    break

    except (requests.RequestException, ValueError) as exc:
        result = _failure(
            str(exc),
            radius_km,
            start_time,
            min_magnitude,
        )
        result["events"] = events
        result["count"] = len(events)
        result["partial"] = bool(events)
        return result

    return {
        "status": "SUCCESS",
        "source": SOURCE_NAME,
        "latitude": latitude,
        "longitude": longitude,
        "radius_km": radius_km,
        "start_time": start_time,
        "min_magnitude": min_magnitude,
        "count": len(events),
        "events": events,
        "complete": not truncated,
        "truncated": truncated,
        "partial": False,
        "note": (
            "Configured event limit reached; additional matching events "
            "may exist."
            if truncated
            else "All pages returned for the specified query were retrieved."
        ),
    }
