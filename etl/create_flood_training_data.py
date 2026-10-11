from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from sklearn.neighbors import BallTree
from urllib3.util.retry import Retry


# ============================================================
# NEXORA — FLOOD TRAINING DATA BUILDER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "climate_cache"

FLOOD_FILE = DATA_DIR / "NEXORA_Flood_Events_Clean.csv"
OUTPUT_FILE = DATA_DIR / "flood_training_dataset_v2.csv"

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

RANDOM_SEED = 42

NEGATIVE_RATIO = 1.0
NEGATIVE_MIN_DISTANCE_KM = 50.0
NEGATIVE_MAX_DISTANCE_KM = 300.0

REQUEST_TIMEOUT = (10, 60)

CLIMATE_BUFFER_DAYS = 7

LOGGER = logging.getLogger("nexora.training")


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ============================================================
# HTTP SESSION
# ============================================================

def create_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=10,
        pool_maxsize=10,
    )

    session = requests.Session()

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": "NEXORA/1.0 flood-training-pipeline",
            "Accept": "application/json",
        }
    )

    return session


# ============================================================
# VALIDATION
# ============================================================

def validate_coordinates(
    latitude: float,
    longitude: float,
) -> None:

    if not -90 <= latitude <= 90:
        raise ValueError(
            f"Invalid latitude: {latitude}"
        )

    if not -180 <= longitude <= 180:
        raise ValueError(
            f"Invalid longitude: {longitude}"
        )


def validate_date_range(
    start_date: str,
    end_date: str,
) -> None:

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if pd.isna(start) or pd.isna(end):
        raise ValueError(
            f"Invalid date range: {start_date} → {end_date}"
        )

    if start > end:
        raise ValueError(
            f"Start date is after end date: "
            f"{start_date} → {end_date}"
        )


# ============================================================
# CACHE
# ============================================================

def cache_key(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> str:

    raw = (
        f"{latitude:.5f}|"
        f"{longitude:.5f}|"
        f"{start_date}|"
        f"{end_date}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def cache_path(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> Path:

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return CACHE_DIR / (
        cache_key(
            latitude,
            longitude,
            start_date,
            end_date,
        )
        + ".json"
    )


# ============================================================
# CLIMATE EXTRACTION
# ============================================================

def fetch_climate(
    session: requests.Session,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:

    validate_coordinates(
        latitude,
        longitude,
    )

    validate_date_range(
        start_date,
        end_date,
    )

    cached_file = cache_path(
        latitude,
        longitude,
        start_date,
        end_date,
    )

    if cached_file.exists():

        try:

            with cached_file.open(
                "r",
                encoding="utf-8",
            ) as file:

                return json.load(file)

        except (
            OSError,
            json.JSONDecodeError,
        ):

            LOGGER.warning(
                "Invalid cache removed: %s",
                cached_file,
            )

            cached_file.unlink(
                missing_ok=True
            )

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

    response = session.get(
        OPEN_METEO_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    payload = response.json()

    if "daily" not in payload:
        raise RuntimeError(
            "Open-Meteo response has no daily data."
        )

    daily = payload["daily"]

    required = [
        "time",
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
    ]

    missing = [
        column
        for column in required
        if column not in daily
    ]

    if missing:
        raise RuntimeError(
            f"Missing climate fields: {missing}"
        )

    with cached_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
        )

    return payload


# ============================================================
# CLIMATE FEATURES
# ============================================================

def climate_features(
    payload: dict[str, Any],
) -> dict[str, float]:

    daily = payload["daily"]

    frame = pd.DataFrame(
        {
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

    frame = frame.dropna()

    if frame.empty:
        raise RuntimeError(
            "No valid climate observations available."
        )

    precipitation = frame["precipitation"]

    return {
        "temperature_mean": float(
            frame["temperature_mean"].mean()
        ),

        "temperature_max": float(
            frame["temperature_max"].max()
        ),

        "temperature_min": float(
            frame["temperature_min"].min()
        ),

        "precipitation_sum": float(
            precipitation.sum()
        ),

        "precipitation_mean": float(
            precipitation.mean()
        ),

        "precipitation_max": float(
            precipitation.max()
        ),

        "precipitation_p95": float(
            precipitation.quantile(0.95)
        ),

        "heavy_rain_days": float(
            (precipitation >= 20).sum()
        ),

        "extreme_rain_days": float(
            (precipitation >= 50).sum()
        ),

        "observation_count": float(
            len(frame)
        ),
    }


# ============================================================
# EVENT PERIOD
# ============================================================

def event_period(
    row: pd.Series,
) -> tuple[str, str]:

    start = pd.to_datetime(
        row["start_date"],
        errors="coerce",
    )

    end = pd.to_datetime(
        row["end_date"],
        errors="coerce",
    )

    if pd.isna(start):

        raise ValueError(
            f"Invalid event start date for "
            f"event {row.get('event_id')}"
        )

    if pd.isna(end):

        duration = row.get(
            "duration_days",
            1,
        )

        duration = (
            int(duration)
            if pd.notna(duration)
            else 1
        )

        end = start + pd.Timedelta(
            days=max(duration - 1, 0)
        )

    # A short buffer captures the rainfall immediately
    # preceding and surrounding the documented event.
    buffered_start = (
        start - pd.Timedelta(
            days=CLIMATE_BUFFER_DAYS
        )
    )

    buffered_end = (
        end + pd.Timedelta(
            days=CLIMATE_BUFFER_DAYS
        )
    )

    return (
        buffered_start.strftime("%Y-%m-%d"),
        buffered_end.strftime("%Y-%m-%d"),
    )


# ============================================================
# DISTANCE
# ============================================================

def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:

    earth_radius_km = 6371.0088

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    delta_lat = math.radians(
        lat2 - lat1
    )

    delta_lon = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(delta_lat / 2) ** 2
        +
        math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(delta_lon / 2) ** 2
    )

    return (
        earth_radius_km
        * 2
        * math.atan2(
            math.sqrt(a),
            math.sqrt(1 - a),
        )
    )


# ============================================================
# NEGATIVE SAMPLE GENERATION
# ============================================================

def generate_negative_samples(
    floods: pd.DataFrame,
    count: int,
) -> pd.DataFrame:

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    flood_coordinates = np.radians(
        floods[
            [
                "latitude",
                "longitude",
            ]
        ].to_numpy()
    )

    tree = BallTree(
        flood_coordinates,
        metric="haversine",
    )

    lat_min = max(
        -89.0,
        float(floods["latitude"].min()) - 5,
    )

    lat_max = min(
        89.0,
        float(floods["latitude"].max()) + 5,
    )

    lon_min = float(
        floods["longitude"].min()
    ) - 5

    lon_max = float(
        floods["longitude"].max()
    ) + 5

    candidates: list[dict[str, Any]] = []

    attempts = 0
    max_attempts = count * 100

    while (
        len(candidates) < count
        and attempts < max_attempts
    ):

        attempts += 1

        latitude = float(
            rng.uniform(
                lat_min,
                lat_max,
            )
        )

        longitude = float(
            rng.uniform(
                lon_min,
                lon_max,
            )
        )

        point = np.radians(
            [[latitude, longitude]]
        )

        distance_rad = tree.query(
            point,
            k=1,
        )[0][0][0]

        distance_km = (
            distance_rad * 6371.0088
        )

        if (
            distance_km
            < NEGATIVE_MIN_DISTANCE_KM
        ):
            continue

        if (
            distance_km
            > NEGATIVE_MAX_DISTANCE_KM
        ):
            continue

        reference_index = int(
            rng.integers(
                0,
                len(floods),
            )
        )

        reference = floods.iloc[
            reference_index
        ]

        candidates.append(
            {
                "event_id": (
                    f"proxy_{len(candidates) + 1}"
                ),
                "latitude": latitude,
                "longitude": longitude,
                "severity": 0.0,
                "deaths": 0.0,
                "displaced": 0.0,
                "duration_days": 0.0,
                "start_date": reference[
                    "start_date"
                ],
                "end_date": reference[
                    "end_date"
                ],
                "cause": "proxy_non_flood",
                "country": "unknown",
                "country_code": "unknown",
                "flood_label": 0,
                "sample_type": "proxy_non_flood",
            }
        )

    if len(candidates) < count:

        raise RuntimeError(
            "Unable to generate the requested "
            "number of spatially separated "
            "proxy non-flood samples."
        )

    return pd.DataFrame(
        candidates
    )


# ============================================================
# PREPARE SOURCE DATA
# ============================================================

def load_flood_events() -> pd.DataFrame:

    if not FLOOD_FILE.exists():

        raise FileNotFoundError(
            f"Flood dataset not found: "
            f"{FLOOD_FILE}"
        )

    floods = pd.read_csv(
        FLOOD_FILE
    )

    required_columns = {
        "event_id",
        "latitude",
        "longitude",
        "start_date",
        "end_date",
        "duration_days",
        "severity",
        "deaths",
        "displaced",
    }

    missing = (
        required_columns
        - set(floods.columns)
    )

    if missing:

        raise ValueError(
            "Flood dataset is missing "
            f"columns: {sorted(missing)}"
        )

    floods = floods.copy()

    numeric_columns = [
        "latitude",
        "longitude",
        "duration_days",
        "severity",
        "deaths",
        "displaced",
    ]

    for column in numeric_columns:

        floods[column] = pd.to_numeric(
            floods[column],
            errors="coerce",
        )

    floods["start_date"] = pd.to_datetime(
        floods["start_date"],
        errors="coerce",
    )

    floods["end_date"] = pd.to_datetime(
        floods["end_date"],
        errors="coerce",
    )

    floods = floods.dropna(
        subset=[
            "latitude",
            "longitude",
            "start_date",
        ]
    )

    floods["end_date"] = floods[
        "end_date"
    ].fillna(
        floods["start_date"]
        + pd.to_timedelta(
            floods["duration_days"]
            .fillna(1)
            .clip(lower=1)
            - 1,
            unit="D",
        )
    )

    floods["flood_label"] = 1
    floods["sample_type"] = (
        "historical_flood"
    )

    floods["start_date"] = (
        floods["start_date"]
        .dt.strftime("%Y-%m-%d")
    )

    floods["end_date"] = (
        floods["end_date"]
        .dt.strftime("%Y-%m-%d")
    )

    floods = floods.drop_duplicates(
        subset=[
            "event_id"
        ]
    )

    return floods.reset_index(
        drop=True
    )


# ============================================================
# BUILD FEATURES
# ============================================================

def build_dataset(
    samples: pd.DataFrame,
) -> pd.DataFrame:

    session = create_session()

    records: list[dict[str, Any]] = []

    total = len(samples)

    try:

        for index, row in samples.iterrows():

            latitude = float(
                row["latitude"]
            )

            longitude = float(
                row["longitude"]
            )

            start_date, end_date = (
                event_period(row)
            )

            LOGGER.info(
                "Climate %s/%s | lat=%.5f lon=%.5f | %s → %s",
                index + 1,
                total,
                latitude,
                longitude,
                start_date,
                end_date,
            )

            try:

                payload = fetch_climate(
                    session=session,
                    latitude=latitude,
                    longitude=longitude,
                    start_date=start_date,
                    end_date=end_date,
                )

                features = climate_features(
                    payload
                )

                record = {
                    "event_id": row[
                        "event_id"
                    ],

                    "latitude": latitude,
                    "longitude": longitude,

                    "severity": float(
                        row.get(
                            "severity",
                            0,
                        )
                    ),

                    "deaths": float(
                        row.get(
                            "deaths",
                            0,
                        )
                    ),

                    "displaced": float(
                        row.get(
                            "displaced",
                            0,
                        )
                    ),

                    "duration_days": float(
                        row.get(
                            "duration_days",
                            0,
                        )
                    ),

                    "start_date": row[
                        "start_date"
                    ],

                    "end_date": row[
                        "end_date"
                    ],

                    "flood_label": int(
                        row["flood_label"]
                    ),

                    "sample_type": row[
                        "sample_type"
                    ],

                    **features,
                }

                records.append(record)

            except Exception as exc:

                LOGGER.error(
                    "Skipping sample %s: %s",
                    row["event_id"],
                    exc,
                )

            # Respect external API limits.
            time.sleep(0.15)

    finally:

        session.close()

    if not records:

        raise RuntimeError(
            "No training records were created."
        )

    return pd.DataFrame(records)


# ============================================================
# QUALITY CONTROL
# ============================================================

def validate_training_dataset(
    dataset: pd.DataFrame,
) -> None:

    if dataset.empty:

        raise ValueError(
            "Training dataset is empty."
        )

    if dataset["flood_label"].nunique() < 2:

        raise ValueError(
            "Training dataset must contain "
            "both flood and non-flood labels."
        )

    feature_columns = [
        "temperature_mean",
        "temperature_max",
        "temperature_min",
        "precipitation_sum",
        "precipitation_mean",
        "precipitation_max",
        "precipitation_p95",
        "heavy_rain_days",
        "extreme_rain_days",
    ]

    missing = [
        column
        for column in feature_columns
        if column not in dataset.columns
    ]

    if missing:

        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    if dataset[
        feature_columns
    ].isna().any().any():

        raise ValueError(
            "Training dataset contains "
            "missing feature values."
        )

    if np.isinf(
        dataset[
            feature_columns
        ].to_numpy()
    ).any():

        raise ValueError(
            "Training dataset contains "
            "infinite feature values."
        )

    duplicate_count = dataset.duplicated(
        subset=[
            "latitude",
            "longitude",
            "start_date",
            "flood_label",
        ]
    ).sum()

    if duplicate_count:

        LOGGER.warning(
            "Detected %s duplicate spatial/time samples.",
            duplicate_count,
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    LOGGER.info(
        "Loading historical flood events..."
    )

    floods = load_flood_events()

    LOGGER.info(
        "Historical flood events: %s",
        len(floods),
    )

    negative_count = int(
        round(
            len(floods)
            * NEGATIVE_RATIO
        )
    )

    LOGGER.info(
        "Generating %s spatial proxy non-flood samples...",
        negative_count,
    )

    negatives = generate_negative_samples(
        floods,
        negative_count,
    )

    samples = pd.concat(
        [
            floods,
            negatives,
        ],
        ignore_index=True,
    )

    samples = samples.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(
        drop=True
    )

    LOGGER.info(
        "Total training samples: %s",
        len(samples),
    )

    dataset = build_dataset(
        samples
    )

    validate_training_dataset(
        dataset
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 60)
    print("NEXORA FLOOD TRAINING DATASET")
    print("=" * 60)

    print(
        f"Total samples: {len(dataset)}"
    )

    print(
        "\nLabel distribution:"
    )

    print(
        dataset[
            "flood_label"
        ].value_counts()
        .sort_index()
    )

    print(
        "\nSample distribution:"
    )

    print(
        dataset[
            "sample_type"
        ].value_counts()
    )

    print(
        "\nClimate feature statistics:"
    )

    print(
        dataset[
            [
                "temperature_mean",
                "temperature_max",
                "temperature_min",
                "precipitation_sum",
                "precipitation_mean",
                "precipitation_max",
                "precipitation_p95",
                "heavy_rain_days",
                "extreme_rain_days",
            ]
        ].describe()
    )

    print(
        f"\nSaved: {OUTPUT_FILE}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Label 0 represents a spatial proxy non-flood "
        "sample, not proof that the location never flooded."
    )


if __name__ == "__main__":
    main()