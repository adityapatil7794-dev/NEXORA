import psycopg2
from math import radians, sin, cos, sqrt, atan2


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "nexora_db",
    "user": "adityaanilpatil",
    "password": "aditya3971"
}


def get_flood_risk(latitude, longitude, radius_km=100):

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    query = """
        SELECT
            event_id,
            country,
            latitude,
            longitude,
            start_date,
            end_date,
            cause,
            severity,
            deaths,
            displaced,
            duration_days
        FROM flood_history
    """

    cur.execute(query)
    rows = cur.fetchall()

    events = []

    for row in rows:

        (
            event_id,
            country,
            event_lat,
            event_lon,
            start_date,
            end_date,
            cause,
            severity,
            deaths,
            displaced,
            duration_days
        ) = row

        # Distance calculation
        R = 6371.0

        lat1 = radians(latitude)
        lat2 = radians(event_lat)

        dlat = radians(event_lat - latitude)
        dlon = radians(event_lon - longitude)

        a = (
            sin(dlat / 2) ** 2
            + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        )

        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance_km = R * c

        if distance_km <= radius_km:

            events.append({
                "event_id": event_id,
                "country": country,
                "latitude": event_lat,
                "longitude": event_lon,
                "start_date": str(start_date),
                "end_date": str(end_date),
                "cause": cause,
                "severity": float(severity) if severity is not None else None,
                "deaths": deaths,
                "displaced": displaced,
                "duration_days": duration_days,
                "distance_km": round(distance_km, 2)
            })

    cur.close()
    conn.close()

    # ------------------------------------------------
    # NO HISTORICAL EVENT = DATA LIMITED
    # ------------------------------------------------

    if len(events) == 0:

        return {
            "historical_events_found": 0,
            "flood_risk_percent": None,
            "risk_level": "DATA LIMITED",
            "confidence": "LOW",
            "reasons": [
                "No recorded flood events were found within the selected radius.",
                "This does NOT mean the site is flood-safe.",
                "The current historical dataset does not contain sufficient local evidence."
            ],
            "events": []
        }

    # ------------------------------------------------
    # RISK CALCULATION
    # ------------------------------------------------

    event_count = len(events)

    max_severity = max(
        event["severity"]
        for event in events
        if event["severity"] is not None
    )

    nearest_distance = min(
        event["distance_km"]
        for event in events
    )

    # Base risk from historical event count
    risk = min(event_count * 20, 60)

    # Severity contribution
    if max_severity >= 2:
        risk += 20
    elif max_severity >= 1.5:
        risk += 15
    else:
        risk += 10

    # Distance contribution
    if nearest_distance <= 25:
        risk += 20
    elif nearest_distance <= 50:
        risk += 10

    risk = min(int(risk), 100)

    if risk >= 70:
        risk_level = "HIGH"
    elif risk >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    if event_count >= 5:
        confidence = "HIGH"
    elif event_count >= 2:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    reasons = [
        f"{event_count} historical flood event(s) found within {radius_km} km.",
        f"Nearest historical event: {nearest_distance:.2f} km away.",
        f"Maximum historical severity: {max_severity}."
    ]

    return {
        "historical_events_found": event_count,
        "flood_risk_percent": risk,
        "risk_level": risk_level,
        "confidence": confidence,
        "reasons": reasons,
        "events": events
    }