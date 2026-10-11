import pandas as pd
import psycopg2
import os

df = pd.read_csv("data/NEXORA_Flood_Events_Clean.csv")

conn = psycopg2.connect(
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", "5432"),
    database=os.getenv("DB_NAME", "nexora_db"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS flood_history (
    event_id INTEGER PRIMARY KEY,
    country_code TEXT,
    country TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    start_date DATE,
    end_date DATE,
    cause TEXT,
    severity DOUBLE PRECISION,
    deaths DOUBLE PRECISION,
    displaced DOUBLE PRECISION,
    duration_days INTEGER
);
""")

for _, r in df.iterrows():
    cur.execute("""
        INSERT INTO flood_history (
            event_id, country_code, country,
            latitude, longitude,
            start_date, end_date,
            cause, severity, deaths,
            displaced, duration_days
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (event_id) DO UPDATE SET
            country = EXCLUDED.country,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude,
            start_date = EXCLUDED.start_date,
            end_date = EXCLUDED.end_date,
            severity = EXCLUDED.severity,
            deaths = EXCLUDED.deaths,
            displaced = EXCLUDED.displaced,
            duration_days = EXCLUDED.duration_days;
    """, tuple(r[
        [
            "event_id", "country_code", "country",
            "latitude", "longitude",
            "start_date", "end_date",
            "cause", "severity", "deaths",
            "displaced", "duration_days"
        ]
    ]))

conn.commit()

print("Flood history loaded:", len(df), "events")

cur.close()
conn.close()