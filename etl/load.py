import logging
import os
from typing import Any

import psycopg2
from psycopg2.extras import execute_batch

from extract import extract_climate_data
from transform import transform_climate_data
from validate import validate_climate_data


logger = logging.getLogger(__name__)


class ClimateLoadError(Exception):
    """Raised when climate data cannot be loaded into PostgreSQL."""


def get_database_connection():
    """Create a secure PostgreSQL connection."""

    try:
        connection = psycopg2.connect(
            host=os.getenv("NEXORA_DB_HOST", "localhost"),
            port=os.getenv("NEXORA_DB_PORT", "5432"),
            database=os.getenv("NEXORA_DB_NAME", "nexora_db"),
            user=os.getenv("NEXORA_DB_USER"),
            password=os.getenv("NEXORA_DB_PASSWORD"),
        )

        return connection

    except psycopg2.Error as exc:
        raise ClimateLoadError(
            "Unable to connect to PostgreSQL."
        ) from exc


def load_climate_data(
    records: list[dict[str, Any]],
    latitude: float,
    longitude: float,
    elevation: float | None,
) -> int:
    """Load validated climate records into PostgreSQL."""

    if not records:
        logger.warning("No climate records to load.")
        return 0

    connection = get_database_connection()

    query = """
        INSERT INTO climate_observations (
            latitude,
            longitude,
            elevation_m,
            observation_date,
            temperature_mean_c,
            temperature_max_c,
            temperature_min_c,
            precipitation_mm,
            data_source,
            source_reference
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
        ON CONFLICT (
            latitude,
            longitude,
            observation_date,
            data_source
        )
        DO UPDATE SET
            elevation_m = EXCLUDED.elevation_m,
            temperature_mean_c = EXCLUDED.temperature_mean_c,
            temperature_max_c = EXCLUDED.temperature_max_c,
            temperature_min_c = EXCLUDED.temperature_min_c,
            precipitation_mm = EXCLUDED.precipitation_mm,
            source_reference = EXCLUDED.source_reference,
            loaded_at = CURRENT_TIMESTAMP;
    """

    values = [
        (
            latitude,
            longitude,
            elevation,
            record["observation_date"],
            record["temperature_mean_c"],
            record["temperature_max_c"],
            record["temperature_min_c"],
            record["precipitation_mm"],
            "Open-Meteo",
            "https://open-meteo.com/",
        )
        for record in records
    ]

    try:

        with connection:

            with connection.cursor() as cursor:

                execute_batch(
                    cursor,
                    query,
                    values,
                    page_size=500,
                )

        logger.info(
            "Loaded %d climate records into PostgreSQL.",
            len(records),
        )

        return len(records)

    except psycopg2.Error as exc:

        connection.rollback()

        raise ClimateLoadError(
            "Failed to load climate records into PostgreSQL."
        ) from exc

    finally:
        connection.close()


def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # -----------------------------
    # Location
    # -----------------------------

    latitude = 18.5204
    longitude = 73.8567

    # -----------------------------
    # Extract
    # -----------------------------

    climate_data = extract_climate_data(
        latitude=latitude,
        longitude=longitude,
        start_date="2025-01-01",
        end_date="2025-12-31",
    )

    # -----------------------------
    # Transform
    # -----------------------------

    transformed_records = transform_climate_data(
        climate_data
    )

    # -----------------------------
    # Validate
    # -----------------------------

    valid_records, invalid_records = validate_climate_data(
        transformed_records
    )

    if invalid_records:
        logger.warning(
            "%d invalid records were excluded.",
            len(invalid_records),
        )

    # -----------------------------
    # Load
    # -----------------------------

    loaded_count = load_climate_data(
        records=valid_records,
        latitude=climate_data["latitude"],
        longitude=climate_data["longitude"],
        elevation=climate_data.get("elevation"),
    )

    print("\nNEXORA climate ETL completed")
    print(f"Extracted: {len(transformed_records)}")
    print(f"Valid: {len(valid_records)}")
    print(f"Invalid: {len(invalid_records)}")
    print(f"Loaded: {loaded_count}")


if __name__ == "__main__":
    main()