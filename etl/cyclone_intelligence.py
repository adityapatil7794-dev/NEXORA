
"""NEXORA cyclone and tropical cyclone historical intelligence.

Source: NOAA IBTrACS (International Best Track Archive for
Climate Stewardship), a global archive of tropical cyclone tracks.

This module retrieves cyclone records from the public NOAA CSV archive.
It reports historical proximity, not a forecast of future cyclones.
"""

import csv
import io
import math
from datetime import datetime, timezone

import requests


NOAA_IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
)

REQUEST_TIMEOUT = 45
MAX_RADIUS_KM = 2000
EARTH_RADIUS_KM = 6371.0088


def _distance_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance using the haversine formula."""
    lat1, lon1, lat2, lon2 = map(
        math.radians, (lat1, lon1, lat2, lon2)
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS_KM * math.asin(
        min(1.0, math.sqrt(a))
    )


def _number(value):
    """Convert a valid CSV number to float, otherwise return None."""
    if value is None:
        return None

    value = value.strip()

    if not value or value.upper() in {"NA", "NAN", "NULL"}:
        return None

    try:
        result = float(value)
    except ValueError:
        return None

    return result if math.isfinite(result) else None


def get_historical_cyclones(
    latitude: float,
    longitude: float,
    radius_km: float = 300,
    start_year: int = 1980,
    end_year: int | None = None,
):
    """Find historical tropical cyclone track points near a site.

    Returns one record per matching track point. Multiple records can
    belong to the same storm.

    The NOAA archive is large, so this function downloads the CSV before
    filtering it. For production use, cache the archive or use a local
    indexed copy rather than downloading it for every site assessment.
    """

    for name, value in (
        ("latitude", latitude),
        ("longitude", longitude),
        ("radius_km", radius_km),
    ):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be numeric.")

        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric.") from exc

        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")

        if name == "latitude":
            latitude = value
        elif name == "longitude":
            longitude = value
        else:
            radius_km = value

    if not -90 <= latitude <= 90:
        raise ValueError("latitude must be between -90 and 90.")

    if not -180 <= longitude <= 180:
        raise ValueError("longitude must be between -180 and 180.")

    if not 0 < radius_km <= MAX_RADIUS_KM:
        raise ValueError(
            f"radius_km must be greater than 0 and at most {MAX_RADIUS_KM}."
        )

    current_year = datetime.now(timezone.utc).year

    if end_year is None:
        end_year = current_year

    if (
        isinstance(start_year, bool)
        or not isinstance(start_year, int)
        or isinstance(end_year, bool)
        or not isinstance(end_year, int)
    ):
        raise ValueError("start_year and end_year must be integers.")

    if not 1840 <= start_year <= end_year <= current_year:
        raise ValueError("Invalid year range.")

    try:
        response = requests.get(
            NOAA_IBTRACS_URL,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "NEXORA-Site-Intelligence/1.0"},
        )
        response.raise_for_status()

        # IBTrACS begins with a metadata row before the CSV header.
        lines = response.text.splitlines()

        if len(lines) < 3:
            raise ValueError("NOAA returned an empty or invalid archive.")

        reader = csv.DictReader(io.StringIO(response.text))
        next(reader, None)  # Skip the units row, not the header

        if not reader.fieldnames or not {
            "SID", "SEASON", "ISO_TIME", "LAT", "LON"
        }.issubset(reader.fieldnames):
            raise ValueError("NOAA archive is missing expected columns.")

        events = []
        storms = set()
        closest_distance = None

        for row in reader:
            try:
                year = int(row.get("SEASON", "").strip())
            except (ValueError, AttributeError):
                continue

            if not start_year <= year <= end_year:
                continue

            storm_lat = _number(row.get("LAT"))
            storm_lon = _number(row.get("LON"))

            if storm_lat is None or storm_lon is None:
                continue

            if not -90 <= storm_lat <= 90 or not -180 <= storm_lon <= 360:
                continue

            # IBTrACS may express longitudes in the 0–360 range.
            storm_lon = ((storm_lon + 180) % 360) - 180

            distance = _distance_km(
                latitude, longitude, storm_lat, storm_lon
            )

            if distance > radius_km:
                continue

            storm_id = (row.get("SID") or "").strip()

            if not storm_id:
                continue

            storms.add(storm_id)

            if closest_distance is None or distance < closest_distance:
                closest_distance = distance

            events.append({
                "event_id": storm_id,
                "storm_name": (row.get("NAME") or "").strip() or None,
                "season": year,
                "time": (row.get("ISO_TIME") or "").strip() or None,
                "latitude": storm_lat,
                "longitude": storm_lon,
                "distance_km": round(distance, 2),
                "maximum_wind_kt": _number(row.get("USA_WIND")),
                "minimum_pressure_mb": _number(row.get("USA_PRES")),
                "nature": (row.get("NATURE") or "").strip() or None,
                "source": "NOAA IBTrACS",
            })

        events.sort(key=lambda event: event["distance_km"])

        return {
            "status": "SUCCESS",
            "source": "NOAA IBTrACS",
            "latitude": latitude,
            "longitude": longitude,
            "radius_km": radius_km,
            "start_year": start_year,
            "end_year": end_year,
            "count": len(events),
            "unique_storm_count": len(storms),
            "closest_distance_km": (
                round(closest_distance, 2)
                if closest_distance is not None
                else None
            ),
            "events": events,
            "note": (
                "Historical storm track points within the selected radius. "
                "These records do not represent a cyclone forecast or "
                "a complete assessment of local wind, rain, or storm surge."
            ),
        }

    except (requests.RequestException, ValueError, csv.Error) as exc:
        return {
            "status": "DATA UNAVAILABLE",
            "source": "NOAA IBTrACS",
            "latitude": latitude,
            "longitude": longitude,
            "radius_km": radius_km,
            "start_year": start_year,
            "end_year": end_year,
            "count": 0,
            "unique_storm_count": 0,
            "closest_distance_km": None,
            "events": [],
            "error": str(exc),
        }
