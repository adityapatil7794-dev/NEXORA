import logging
from datetime import date, datetime
from typing import Any

from extract import extract_climate_data
from transform import transform_climate_data


logger = logging.getLogger(__name__)


class ClimateValidationError(Exception):
    """Raised when climate validation cannot be completed."""


def normalize_date(value: Any) -> date | None:
    """Convert supported date formats into a Python date."""

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None

    return None


def validate_climate_data(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    if not isinstance(records, list):
        raise ClimateValidationError(
            "Climate records must be a list."
        )

    valid_records = []
    invalid_records = []
    seen_dates: set[date] = set()

    for index, record in enumerate(records):

        errors = []

        observation_date = normalize_date(
            record.get("observation_date")
        )

        mean_temp = record.get("temperature_mean_c")
        max_temp = record.get("temperature_max_c")
        min_temp = record.get("temperature_min_c")
        precipitation = record.get("precipitation_mm")

        # Date validation
        if observation_date is None:
            errors.append("Invalid observation date")

        elif observation_date in seen_dates:
            errors.append("Duplicate observation date")

        else:
            seen_dates.add(observation_date)

        # Missing values
        if mean_temp is None:
            errors.append("Missing mean temperature")

        if max_temp is None:
            errors.append("Missing maximum temperature")

        if min_temp is None:
            errors.append("Missing minimum temperature")

        if precipitation is None:
            errors.append("Missing precipitation")

        # Numeric validation
        numeric_fields = {
            "temperature_mean_c": mean_temp,
            "temperature_max_c": max_temp,
            "temperature_min_c": min_temp,
            "precipitation_mm": precipitation,
        }

        for field, value in numeric_fields.items():

            if value is not None and not isinstance(
                value,
                (int, float),
            ):
                errors.append(f"{field} is not numeric")

        # Temperature consistency
        if (
            isinstance(min_temp, (int, float))
            and isinstance(max_temp, (int, float))
            and min_temp > max_temp
        ):
            errors.append(
                "Minimum temperature is greater than maximum temperature"
            )

        # Temperature sanity check
        for temperature in (
            mean_temp,
            max_temp,
            min_temp,
        ):
            if (
                isinstance(temperature, (int, float))
                and not -90 <= temperature <= 70
            ):
                errors.append(
                    "Temperature outside expected physical range"
                )
                break

        # Precipitation cannot be negative
        if (
            isinstance(precipitation, (int, float))
            and precipitation < 0
        ):
            errors.append("Negative precipitation")

        # Normalize the record
        if observation_date is not None:
            record["observation_date"] = observation_date

        if errors:
            invalid_records.append({
                "record_index": index,
                "record": record,
                "errors": errors,
            })
        else:
            valid_records.append(record)

    logger.info(
        "Climate validation completed: %d valid, %d invalid.",
        len(valid_records),
        len(invalid_records),
    )

    return valid_records, invalid_records


def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    climate_data = extract_climate_data(
        latitude=18.5204,
        longitude=73.8567,
        start_date="2025-01-01",
        end_date="2025-12-31",
    )

    transformed_records = transform_climate_data(
        climate_data
    )

    valid_records, invalid_records = validate_climate_data(
        transformed_records
    )

    print("\nClimate validation completed")
    print(f"Total records: {len(transformed_records)}")
    print(f"Valid records: {len(valid_records)}")
    print(f"Invalid records: {len(invalid_records)}")

    if invalid_records:
        print("\nFirst validation error:")
        print(invalid_records[0])
    else:
        print("\nAll climate records passed validation.")


if __name__ == "__main__":
    main()