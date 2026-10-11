import logging
from typing import Any

from extract import extract_climate_data

logger = logging.getLogger(__name__)


class ClimateTransformError(Exception):
    """Raised when climate data transformation fails."""


def transform_climate_data(
    data: dict[str, Any],
) -> list[dict[str, Any]]:

    daily = data.get("daily")

    if not isinstance(daily, dict):
        raise ClimateTransformError("Missing daily climate data.")

    required_fields = (
        "time",
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
    )

    for field in required_fields:
        if field not in daily:
            raise ClimateTransformError(
                f"Missing required field: {field}"
            )

    dates = daily["time"]

    arrays = [
        daily["temperature_2m_mean"],
        daily["temperature_2m_max"],
        daily["temperature_2m_min"],
        daily["precipitation_sum"],
    ]

    if any(len(array) != len(dates) for array in arrays):
        raise ClimateTransformError(
            "Climate data arrays have inconsistent lengths."
        )

    records = []

    for index, observation_date in enumerate(dates):

        records.append({
            "observation_date": observation_date,
            "temperature_mean_c": daily["temperature_2m_mean"][index],
            "temperature_max_c": daily["temperature_2m_max"][index],
            "temperature_min_c": daily["temperature_2m_min"][index],
            "precipitation_mm": daily["precipitation_sum"][index],
        })

    logger.info(
        "Transformed %d climate records.",
        len(records),
    )

    return records


if __name__ == "__main__":

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

    records = transform_climate_data(climate_data)

    print("\nClimate transformation successful")
    print(f"Records transformed: {len(records)}")

    if records:
        print("First record:")
        print(records[0])