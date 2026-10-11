import pandas as pd
from pathlib import Path

INPUT = Path("data/NEXORA_Flood_Events.csv")
OUTPUT = Path("data/NEXORA_Flood_Events_Clean.csv")

# Load
df = pd.read_csv(INPUT)

# Remove unnecessary columns
df = df.drop(columns=[".geo", "system:index"], errors="ignore")

# Convert dates
df["start_date"] = pd.to_datetime(
    df["start_date"].astype(str), format="%Y%m%d", errors="coerce"
)

df["end_date"] = pd.to_datetime(
    df["end_date"].astype(str), format="%Y%m%d", errors="coerce"
)

# Create event duration
df["duration_days"] = (
    df["end_date"] - df["start_date"]
).dt.days + 1

# Numeric columns
numeric_cols = [
    "event_id",
    "latitude",
    "longitude",
    "severity",
    "deaths",
    "displaced",
    "duration_days"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing country values
df["country"] = df["country"].fillna("Unknown")
df["country_code"] = df["country_code"].fillna("Unknown")
df["cause"] = df["cause"].fillna("Unknown")

# Remove rows without essential location/date
df = df.dropna(
    subset=["event_id", "latitude", "longitude", "start_date", "end_date"]
)

# Remove duplicate events
df = df.drop_duplicates(subset="event_id")

# Save
df.to_csv(OUTPUT, index=False)

print("========== FLOOD DATA PREPARED ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Saved:", OUTPUT)
print("Missing values:")
print(df.isnull().sum())
print("==========================================")