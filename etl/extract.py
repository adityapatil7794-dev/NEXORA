from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LOGGER = logging.getLogger("nexora.climate")

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

REQUEST_TIMEOUT = (10, 60)

RETRY_STRATEGY = Retry(
    total=4,
    connect=4,
    read=4,
    status=4,
    backoff_factor=1.0,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset({"GET"}),
    respect_retry_after_header=True,
)


@dataclass(frozen=True)
class ClimateExtractionConfig:
    latitude: float
    longitude: float
    start_date: str
    end_date: str


def _validate_coordinates(latitude: float, longitude: float) -> None:
    if not -90.0 <= latitude <= 90.0:
        raise ValueError("Latitude must be between -90 and 90.")

    if not -180.0 <= longitude <= 180.0:
        raise ValueError("Longitude must be between -180 and 180.")


def _validate_dates(start_date: str, end_date: str) -> None:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    if start > end:
        raise ValueError("start_date cannot be later than end_date.")


def _create_session() -> requests.Session:
    session = requests.Session()

    adapter = HTTPAdapter(
        max_retries=RETRY_STRATEGY,
        pool_connections=10,
        pool_maxsize=10,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": "NEXORA/1.0 climate-data-pipeline",
            "Accept": "application/json",
        }
    )

    return session


def extract_climate_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """
    Retrieve historical daily climate observations for an arbitrary
    geographic coordinate and date range.

    This function intentionally does not use fixed coordinates,
    fixed dates, or fabricated climate values.
    """

    _validate_coordinates(latitude, longitude)
    _validate_dates(start_date, end_date)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(
            [
                "temperature_2m_mean",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
            ]
        ),
        "timezone": "UTC",
    }

    session = _create_session()

    try:
        LOGGER.info(
            "Requesting climate data: lat=%s lon=%s period=%s to %s",
            latitude,
            longitude,
            start_date,
            end_date,
        )

        response = session.get(
            OPEN_METEO_ARCHIVE_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        payload = response.json()

        if "daily" not in payload:
            raise RuntimeError(
                "Climate API response does not contain daily observations."
            )

        daily = payload["daily"]

        required_fields = {
            "time",
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
        }

        missing_fields = required_fields.difference(daily)

        if missing_fields:
            raise RuntimeError(
                f"Climate API response is missing fields: "
                f"{sorted(missing_fields)}"
            )

        record_count = len(daily["time"])

        if record_count == 0:
            raise RuntimeError("Climate API returned zero observations.")

        for field in required_fields:
            if len(daily[field]) != record_count:
                raise RuntimeError(
                    f"Inconsistent observation length for field: {field}"
                )

        LOGGER.info(
            "Climate extraction successful: %s observations",
            record_count,
        )

        return payload

    except requests.RequestException as exc:
        LOGGER.exception("Climate API request failed.")
        raise RuntimeError(
            "Unable to retrieve historical climate data."
        ) from exc

    except ValueError as exc:
        LOGGER.exception("Climate API returned invalid JSON.")
        raise RuntimeError(
            "Climate API returned invalid JSON."
        ) from exc

    finally:
        session.close()


def climate_dataframe(climate_payload: dict[str, Any]) -> pd.DataFrame:
    """
    Convert the API response into a validated tabular structure.
    """

    daily = climate_payload["daily"]

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"], errors="coerce"),
            "temperature_mean": pd.to_numeric(
                daily["temperature_2m_mean"],
                errors="coerce",
            ),
            "temperature_max": pd.to_numeric(
                daily["temperature_2m_max"],
                errors="coerce",
            ),
            "temperature_min": pd.to_numeric(
                daily["temperature_2m_min"],
                errors="coerce",
            ),
            "precipitation": pd.to_numeric(
                daily["precipitation_sum"],
                errors="coerce",
            ),
        }
    )

    if frame["date"].isna().any():
        raise ValueError("Climate dataset contains invalid dates.")

    numeric_columns = [
        "temperature_mean",
        "temperature_max",
        "temperature_min",
        "precipitation",
    ]

    if frame[numeric_columns].isna().all(axis=1).any():
        raise ValueError(
            "Climate dataset contains rows with no usable measurements."
        )

    if (frame["precipitation"].dropna() < 0).any():
        raise ValueError("Precipitation cannot be negative.")

    frame = frame.sort_values("date").drop_duplicates(
        subset=["date"],
        keep="last",
    )

    return frame.reset_index(drop=True)


def save_climate_csv(
    climate_payload: dict[str, Any],
    output_file: str | Path,
) -> Path:
    """
    Persist extracted climate observations as CSV.
    """

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = climate_dataframe(climate_payload)

    frame.to_csv(output_path, index=False)

    LOGGER.info(
        "Saved %s climate observations to %s",
        len(frame),
        output_path,
    )

    return output_path


def summarize_climate(
    climate_payload: dict[str, Any],
) -> dict[str, float | int]:
    """
    Generate deterministic climate summary features from observations.
    """

    frame = climate_dataframe(climate_payload)

    precipitation = frame["precipitation"]

    return {
        "temperature_mean": round(
            float(frame["temperature_mean"].mean()),
            4,
        ),
        "temperature_max": round(
            float(frame["temperature_max"].max()),
            4,
        ),
        "temperature_min": round(
            float(frame["temperature_min"].min()),
            4,
        ),
        "precipitation_sum": round(
            float(precipitation.sum()),
            4,
        ),
        "precipitation_mean": round(
            float(precipitation.mean()),
            4,
        ),
        "precipitation_max": round(
            float(precipitation.max()),
            4,
        ),
        "observation_count": int(len(frame)),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    latitude = 18.5204
    longitude = 73.8567
    start_date = "2025-01-01"
    end_date = "2025-12-31"

    payload = extract_climate_data(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
    )

    output = save_climate_csv(
        payload,
        Path(__file__).resolve().parent.parent
        / "data"
        / "climate_observations.csv",
    )

    summary = summarize_climate(payload)

    print("\nNEXORA CLIMATE EXTRACTION")
    print("=========================")
    print(f"Latitude: {latitude}")
    print(f"Longitude: {longitude}")
    print(f"Period: {start_date} → {end_date}")
    print(f"Observations: {summary['observation_count']}")
    print(f"Mean temperature: {summary['temperature_mean']} °C")
    print(f"Maximum temperature: {summary['temperature_max']} °C")
    print(f"Minimum temperature: {summary['temperature_min']} °C")
    print(f"Total precipitation: {summary['precipitation_sum']} mm")
    print(f"Maximum daily precipitation: {summary['precipitation_max']} mm")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()