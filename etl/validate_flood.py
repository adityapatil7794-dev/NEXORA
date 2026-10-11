import pandas as pd
from pathlib import Path

# File location
file_path = Path("data/NEXORA_Flood_Events.csv")

# Load dataset
df = pd.read_csv(file_path)

print("========== NEXORA FLOOD DATA ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate event IDs:")
print(df["event_id"].duplicated().sum())

print("\nData types:")
print(df.dtypes)

# Convert dates
df["start_date"] = pd.to_datetime(
    df["start_date"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

df["end_date"] = pd.to_datetime(
    df["end_date"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

# Validate coordinates
print("\nInvalid latitude:",
      ((df["latitude"] < -90) | (df["latitude"] > 90)).sum())

print("Invalid longitude:",
      ((df["longitude"] < -180) | (df["longitude"] > 180)).sum())

print("\nDate range:")
print("From:", df["start_date"].min())
print("To:", df["end_date"].max())

print("\nSeverity distribution:")
print(df["severity"].value_counts(dropna=False))

print("\n========== VALIDATION COMPLETE ==========")