import pandas as pd
from pathlib import Path


# ============================================================
# NEXORA - BUILD FLOOD ML DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FLOOD_FILE = BASE_DIR / "data" / "NEXORA_Flood_Events_Clean.csv"
CLIMATE_FILE = BASE_DIR / "data" / "climate_observations.csv"
OUTPUT_FILE = BASE_DIR / "data" / "flood_ml_dataset.csv"


# ============================================================
# 1. LOAD FLOOD DATA
# ============================================================

print("Loading flood data...")

floods = pd.read_csv(FLOOD_FILE)

print(f"Flood records loaded: {len(floods)}")


# ============================================================
# 2. LOAD CLIMATE DATA
# ============================================================

print("Loading climate data...")

climate = pd.read_csv(CLIMATE_FILE)

print(f"Climate records loaded: {len(climate)}")


# ============================================================
# 3. VALIDATE REQUIRED COLUMNS
# ============================================================

required_flood_columns = [
    "event_id",
    "country",
    "country_code",
    "latitude",
    "longitude",
    "start_date",
    "end_date",
    "cause",
    "severity",
    "deaths",
    "displaced",
    "duration_days"
]

required_climate_columns = [
    "date",
    "temperature_mean",
    "temperature_max",
    "temperature_min",
    "precipitation"
]

for column in required_flood_columns:
    if column not in floods.columns:
        raise ValueError(
            f"Missing flood column: {column}"
        )

for column in required_climate_columns:
    if column not in climate.columns:
        raise ValueError(
            f"Missing climate column: {column}"
        )


# ============================================================
# 4. CONVERT DATA TYPES
# ============================================================

floods["start_date"] = pd.to_datetime(
    floods["start_date"],
    errors="coerce"
)

floods["end_date"] = pd.to_datetime(
    floods["end_date"],
    errors="coerce"
)

climate["date"] = pd.to_datetime(
    climate["date"],
    errors="coerce"
)


# ============================================================
# 5. REMOVE INVALID RECORDS
# ============================================================

floods = floods.dropna(
    subset=[
        "latitude",
        "longitude",
        "severity",
        "start_date"
    ]
)

climate = climate.dropna(
    subset=[
        "temperature_mean",
        "temperature_max",
        "temperature_min",
        "precipitation"
    ]
)


# ============================================================
# 6. CREATE CLIMATE FEATURES
# ============================================================

climate_features = {
    "temperature_mean": climate["temperature_mean"].mean(),
    "temperature_max": climate["temperature_max"].max(),
    "temperature_min": climate["temperature_min"].min(),
    "precipitation_sum": climate["precipitation"].sum(),
    "precipitation_mean": climate["precipitation"].mean(),
    "precipitation_max": climate["precipitation"].max()
}

print("\nClimate features:")
for key, value in climate_features.items():
    print(f"{key}: {value}")


# ============================================================
# 7. BUILD FLOOD FEATURE DATASET
# ============================================================

rows = []

for _, event in floods.iterrows():

    row = {
        # Location
        "latitude": event["latitude"],
        "longitude": event["longitude"],

        # Historical flood information
        "severity": event["severity"],
        "deaths": event["deaths"],
        "displaced": event["displaced"],
        "duration_days": event["duration_days"],

        # Climate features
        "temperature_mean": climate_features[
            "temperature_mean"
        ],

        "temperature_max": climate_features[
            "temperature_max"
        ],

        "temperature_min": climate_features[
            "temperature_min"
        ],

        "precipitation_sum": climate_features[
            "precipitation_sum"
        ],

        "precipitation_mean": climate_features[
            "precipitation_mean"
        ],

        "precipitation_max": climate_features[
            "precipitation_max"
        ],

        # Target
        "flood_label": 1
    }

    rows.append(row)


# ============================================================
# 8. CREATE DATAFRAME
# ============================================================

ml = pd.DataFrame(rows)


# ============================================================
# 9. REMOVE INVALID VALUES
# ============================================================

ml = ml.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

ml = ml.dropna()


# ============================================================
# 10. SAVE ML DATASET
# ============================================================

ml.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("       FLOOD ML DATASET")
print("========================================")

print(f"Flood samples: {len(ml)}")
print(f"Columns: {len(ml.columns)}")

print("\nColumns:")
for column in ml.columns:
    print(f"- {column}")

print("\nDataset preview:")
print(ml.head())

print("\nFlood label distribution:")
print(ml["flood_label"].value_counts())

print("\n========================================")
print(f"Saved: {OUTPUT_FILE}")
print("========================================")