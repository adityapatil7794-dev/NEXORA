import requests
from etl.route_analysis import get_route
from etl.risk_response_intelligence import build_risk_response
from etl.scenario_intelligence import analyze_scenarios
from etl.route_analysis import get_route
from etl.infrastructure_analysis import analyze_infrastructure
from etl.terrain_intelligence import analyze_terrain
from datetime import date, timedelta
from pathlib import Path
import math
import os
from typing import Any, Optional
import pandas as pd
import psycopg2
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from etl.extract import extract_climate_data, climate_dataframe
from ml.predict import predict_flood
app = FastAPI(
    title="NEXORA",
    description="Global Project & Site Intelligence Platform",
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"
ML_DIR = BASE_DIR / "ml"

# NEXORA_HOSTED_DATABASE_V1
if os.getenv("DATABASE_URL"):
    from urllib.parse import urlparse, unquote
    _nx_db = urlparse(os.environ["DATABASE_URL"])
    os.environ["NEXORA_DB_HOST"] = unquote(_nx_db.hostname or "")
    os.environ["NEXORA_DB_PORT"] = str(_nx_db.port or 5432)
    os.environ["NEXORA_DB_NAME"] = unquote(_nx_db.path.lstrip("/"))
    os.environ["NEXORA_DB_USER"] = unquote(_nx_db.username or "")
    os.environ["NEXORA_DB_PASSWORD"] = unquote(_nx_db.password or "")
    os.environ["PGHOST"] = unquote(_nx_db.hostname or "")
    os.environ["PGPORT"] = str(_nx_db.port or 5432)
    os.environ["PGDATABASE"] = unquote(_nx_db.path.lstrip("/"))
    os.environ["PGUSER"] = unquote(_nx_db.username or "")
    os.environ["PGPASSWORD"] = unquote(_nx_db.password or "")
    os.environ["PGSSLMODE"] = "require"

DB_HOST = os.getenv("NEXORA_DB_HOST", "localhost")
DB_PORT = int(os.getenv("NEXORA_DB_PORT", "5432"))
DB_NAME = os.getenv("NEXORA_DB_NAME", "nexora_db")
DB_USER = os.getenv("NEXORA_DB_USER", os.getenv("USER", "postgres"))
DB_PASSWORD = os.getenv("NEXORA_DB_PASSWORD", "")
def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=5,
    )
def validate_coordinates(latitude: float, longitude: float):
    if not math.isfinite(latitude) or not -90 <= latitude <= 90:
        raise HTTPException(
            status_code=400,
            detail="Latitude must be between -90 and 90."
        )
    if not math.isfinite(longitude) or not -180 <= longitude <= 180:
        raise HTTPException(
            status_code=400,
            detail="Longitude must be between -180 and 180."
        )
def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )
    a = min(1.0, max(0.0, a))
    return radius * 2 * math.asin(math.sqrt(a))
def json_value(value: Any):
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)
# ============================================================
# REQUEST MODELS
# ============================================================
class ProjectCreate(BaseModel):
    project_name: str = Field(..., min_length=1, max_length=200)
    project_type: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
class PlotCreate(BaseModel):
    plot_name: str = Field(..., min_length=1, max_length=200)
    boundary: list[list[float]] = Field(..., min_length=4)
    area_sq_km: Optional[float] = Field(default=None, ge=0)
class SoilAnalysisCreate(BaseModel):
    soil_type: Optional[str] = None
    ph: Optional[float] = Field(default=None, ge=0, le=14)
    organic_matter: Optional[float] = Field(default=None, ge=0)
    moisture_percentage: Optional[float] = Field(
        default=None,
        ge=0,
        le=100
    )
    permeability: Optional[float] = Field(default=None, ge=0)
    permeability_unit: Optional[str] = None
    bearing_capacity: Optional[float] = Field(default=None, ge=0)
    bearing_capacity_unit: Optional[str] = None
    data_source: str = "CUSTOMER_PROVIDED"
    source_reference: Optional[str] = None
    data_confidence: str = "MEDIUM"
    notes: Optional[str] = None
# ============================================================


# NEXORA Soil Intelligence: geographic SoilGrids estimates.
@app.get("/api/soil/geographic")
def get_geographic_soil(latitude: float, longitude: float):
    validate_coordinates(latitude, longitude)
    try:
        # Import only when requested so API startup does not depend on GDAL.
        from etl.soil_intelligence import get_soil_dataset

        properties = get_soil_dataset(latitude, longitude)
        available = sum(
            1 for item in properties.values()
            if isinstance(item, dict) and item.get("value") is not None
        )

        return {
            "status": "success" if available else "partial",
            "latitude": latitude,
            "longitude": longitude,
            "source": "ISRIC SoilGrids",
            "depth": "0-5 cm",
            "resolution_m": 250,
            "available_properties": available,
            "property_count": len(properties),
            "interpretation": (
                "Modelled geographic estimates, not laboratory measurements."
            ),
            "properties": properties,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Soil estimate extraction failed: {exc}"
        )

# API
# ============================================================
@app.get("/api")
def api_status():
    return {
        "application": "NEXORA",
        "status": "operational",
        "version": app.version,
        "platform": "Project & Site Intelligence",
        "modules": [
            "location",
            "climate",
            "historical_flood",
            "flood_ml",
            "projects",
            "plots",
            "soil_records",
        ],
    }
@app.get("/api/health")
def health_check():
    database_status = "unavailable"
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            database_status = "connected"
        finally:
            connection.close()
    except Exception:
        database_status = "unavailable"
    model_status = (
        "available"
        if (ML_DIR / "flood_model.pkl").exists()
        else "unavailable"
    )
    return {
        "status": "healthy",
        "database": database_status,
        "flood_ml_model": model_status,
    }
# ============================================================
# LOCATION
# ============================================================
@app.get("/location")
def get_location(
    latitude: float = Query(...),
    longitude: float = Query(...)
):
    validate_coordinates(latitude, longitude)
    return {
        "latitude": latitude,
        "longitude": longitude,
        "status": "valid",
    }
# ============================================================
# CLIMATE
# ============================================================
def get_historical_climate(latitude: float, longitude: float):
    validate_coordinates(latitude, longitude)
    end_date = date.today() - timedelta(days=7)
    start_date = end_date - timedelta(days=365)
    payload = extract_climate_data(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )
    dataframe = climate_dataframe(payload)
    if dataframe.empty:
        raise HTTPException(
            status_code=422,
            detail="No climate observations available."
        )
    required = [
        "temperature_mean",
        "temperature_max",
        "temperature_min",
        "precipitation",
    ]
    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]
    if missing:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Required climate fields are missing.",
                "missing_columns": missing,
            },
        )
    for column in required:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce"
        )
    dataframe = dataframe.dropna(
        subset=required
    ).copy()
    if dataframe.empty:
        raise HTTPException(
            status_code=422,
            detail="Climate observations contain no valid values."
        )
    return dataframe, start_date, end_date
@app.get("/analyze/climate")
def analyze_climate(
    latitude: float = Query(...),
    longitude: float = Query(...)
):
    try:
        dataframe, start_date, end_date = (
            get_historical_climate(
                latitude,
                longitude
            )
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Climate data retrieval failed: {exc}"
        )
    precipitation = dataframe["precipitation"]
    return {
        "status": "success",
        "location": {
            "latitude": latitude,
            "longitude": longitude,
        },
        "period": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        "observation_count": len(dataframe),
        "climate": {
            "temperature_mean": round(
                float(dataframe["temperature_mean"].mean()), 2
            ),
            "temperature_max": round(
                float(dataframe["temperature_max"].max()), 2
            ),
            "temperature_min": round(
                float(dataframe["temperature_min"].min()), 2
            ),
            "precipitation_sum": round(
                float(precipitation.sum()), 2
            ),
            "precipitation_mean": round(
                float(precipitation.mean()), 2
            ),
            "precipitation_max": round(
                float(precipitation.max()), 2
            ),
            "precipitation_p95": round(
                float(precipitation.quantile(0.95)), 2
            ),
            "heavy_rain_days": int(
                (precipitation >= 20).sum()
            ),
            "extreme_rain_days": int(
                (precipitation >= 50).sum()
            ),
        },
        "daily_series": [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "temperature_mean": round(float(row["temperature_mean"]), 2),
                "temperature_max": round(float(row["temperature_max"]), 2),
                "temperature_min": round(float(row["temperature_min"]), 2),
                "precipitation_mm": round(float(row["precipitation"]), 2),
            }
            for _, row in dataframe.iterrows()
        ],
        "source": "Open-Meteo Historical Weather API",
    }
# ============================================================
# HISTORICAL FLOOD
# ============================================================
# ============================================================
# HISTORICAL FLOOD INTELLIGENCE
# ============================================================
@app.get("/api/analyze/flood")
def analyze_flood(
    latitude: float = Query(...),
    longitude: float = Query(...),
    radius_km: float = Query(
        10.0,
        ge=1.0,
        le=500.0,
    ),
):
    validate_coordinates(latitude, longitude)
    query = """
        SELECT
            event_id,
            latitude,
            longitude,
            start_date,
            end_date,
            duration_days,
            country
        FROM flood_history
        WHERE latitude IS NOT NULL
          AND longitude IS NOT NULL
    """
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
        finally:
            connection.close()
    except Exception as exc:
        return {
            "location": {
                "latitude": latitude,
                "longitude": longitude,
            },
            "radius_km": radius_km,
            "historical_events": [],
            "event_count": 0,
            "risk_status": "DATA LIMITED",
            "confidence": "LOW",
            "message": (
                "Historical flood database is unavailable. "
                "No flood-safety conclusion can be made."
            ),
        }
    events = []
    nearest_event = None
    for row in rows:
        (
            event_id,
            event_lat,
            event_lon,
            start_date_value,
            end_date_value,
            duration_days,
            country,
        ) = row
        try:
            distance_km = haversine_km(
                latitude,
                longitude,
                float(event_lat),
                float(event_lon),
            )
        except Exception:
            continue
        event = {
            "event_id": event_id,
            "latitude": float(event_lat),
            "longitude": float(event_lon),
            "distance_km": round(distance_km, 2),
            "start_date": json_value(start_date_value),
            "end_date": json_value(end_date_value),
            "duration_days": duration_days,
            "country": country,
        }
        # Keep the closest historical event
        if (
            nearest_event is None
            or distance_km < nearest_event["distance_km"]
        ):
            nearest_event = event.copy()
        # Keep events inside requested radius
        if distance_km <= radius_km:
            events.append(event)
    events.sort(
        key=lambda event: event["distance_km"]
    )
    historical_status = (
        "EVENTS DETECTED"
        if events
        else "NO MAPPED EVENTS"
    )
    confidence = (
        "MODERATE"
        if events
        else "LOW"
    )
    if events:
        interpretation = (
            "Mapped historical flood events were "
            "identified within the selected radius."
        )
    elif nearest_event:
        interpretation = (
            "No mapped historical flood events were "
            "identified within the selected radius. "
            "The nearest mapped event is outside the "
            "selected radius. Absence of mapped events "
            "does not prove that the location is flood-safe."
        )
    else:
        interpretation = (
            "No usable mapped historical flood events "
            "were available in the dataset. Absence of "
            "mapped events does not prove that the "
            "location is flood-safe."
        )
    return {
        "location": {
            "latitude": latitude,
            "longitude": longitude,
        },
        "radius_km": radius_km,
        "historical_events": events,
        "event_count": len(events),
        "risk_status": historical_status,
        "confidence": confidence,
        "nearest_historical_event": nearest_event,
        "dataset": {
            "name": "Global Flood Database",
            "coverage": "Mapped historical flood events",
        },
        "interpretation": interpretation,
    }
# ============================================================
# FLOOD ML
# ============================================================
@app.post("/api/analyze/flood-ml")
def analyze_flood_ml(
    latitude: float = Query(...),
    longitude: float = Query(...)
):
    validate_coordinates(latitude, longitude)
    try:
        dataframe, start_date, end_date = (
            get_historical_climate(
                latitude,
                longitude
            )
        )
        precipitation = dataframe["precipitation"]
        features = {
            "temperature_mean": float(
                dataframe["temperature_mean"].mean()
            ),
            "temperature_max": float(
                dataframe["temperature_max"].max()
            ),
            "temperature_min": float(
                dataframe["temperature_min"].min()
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
            "heavy_rain_days": int(
                (precipitation >= 20).sum()
            ),
            "extreme_rain_days": int(
                (precipitation >= 50).sum()
            ),
        }
        prediction = predict_flood(
            latitude=latitude,
            longitude=longitude,
            **features
        )
        response = {
            "status": "success",
            "location": {
                "latitude": latitude,
                "longitude": longitude,
            },
            "climate_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
            "observation_count": len(dataframe),
            "features": features,
            "prediction": prediction,
            "ml_prediction": prediction,
            "source": "NEXORA calibrated flood ML model",
            "warning": (
                "Model probability is an analytical signal, "
                "not a certified engineering assessment."
            ),
        }
        return response
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Flood ML analysis failed: {exc}"
        )
# ============================================================
# PROJECTS
# ============================================================
@app.post("/api/projects")
def create_project(payload: ProjectCreate):
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO projects
                    (
                        project_name,
                        project_type,
                        description
                    )
                    VALUES
                    (%s, %s, %s)
                    RETURNING project_id
                    """,
                    (
                        payload.project_name,
                        payload.project_type,
                        payload.description,
                    ),
                )
                project_id = cursor.fetchone()[0]
            connection.commit()
        finally:
            connection.close()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Project creation failed: {exc}"
        )
    return {
        "status": "success",
        "project_id": project_id,
    }
@app.get("/api/projects")
def get_projects():
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        project_id,
                        project_name,
                        project_type,
                        description
                    FROM projects
                    ORDER BY project_id DESC
                    """
                )
                rows = cursor.fetchall()
        finally:
            connection.close()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Project retrieval failed: {exc}"
        )
    return {
        "count": len(rows),
        "projects": [
            {
                "project_id": row[0],
                "project_name": row[1],
                "project_type": row[2],
                "description": row[3],
            }
            for row in rows
        ],
    }
# ============================================================
# PLOTS
# ============================================================
@app.post("/api/projects/{project_id}/plots")
def create_plot(project_id: int, payload: PlotCreate):
    if len(payload.boundary) < 4:
        raise HTTPException(
            status_code=400,
            detail="Boundary must contain at least 4 coordinates."
        )
    # Validate coordinate format
    coordinates = []
    for point in payload.boundary:
        if len(point) != 2:
            raise HTTPException(
                status_code=400,
                detail="Each boundary point must be [longitude, latitude]."
            )
        longitude = float(point[0])
        latitude = float(point[1])
        validate_coordinates(latitude, longitude)
        coordinates.append([longitude, latitude])
    # Polygon must be closed
    if coordinates[0] != coordinates[-1]:
        coordinates.append(coordinates[0])
    # Create WKT polygon
    polygon_coordinates = ", ".join(
        f"{lon} {lat}"
        for lon, lat in coordinates
    )
    polygon_wkt = f"POLYGON(({polygon_coordinates}))"
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                # Verify project
                cursor.execute(
                    """
                    SELECT project_id
                    FROM projects
                    WHERE project_id = %s
                    """,
                    (project_id,)
                )
                if cursor.fetchone() is None:
                    raise HTTPException(
                        status_code=404,
                        detail="Project not found."
                    )
                # Calculate area using PostGIS
                cursor.execute(
                    """
                    INSERT INTO plots
                    (
                        project_id,
                        plot_name,
                        boundary,
                        area_sq_km
                    )
                    VALUES
                    (
                        %s,
                        %s,
                        ST_GeomFromText(%s, 4326),
                        ST_Area(
                            ST_GeomFromText(%s, 4326)::geography
                        ) / 1000000
                    )
                    RETURNING
                        plot_id,
                        project_id,
                        plot_name,
                        area_sq_km,
                        ST_GeometryType(boundary),
                        ST_SRID(boundary)
                    """,
                    (
                        project_id,
                        payload.plot_name,
                        polygon_wkt,
                        polygon_wkt,
                    )
                )
                row = cursor.fetchone()
            connection.commit()
        finally:
            connection.close()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Plot creation failed: {exc}"
        )
    return {
        "status": "success",
        "plot": {
            "plot_id": row[0],
            "project_id": row[1],
            "plot_name": row[2],
            "area_sq_km": float(row[3]),
            "geometry_type": row[4],
            "srid": row[5],
        }
    }
@app.get("/api/projects/{project_id}/plots")
def get_project_plots(project_id: int):
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        plot_id,
                        project_id,
                        plot_name,
                        area_sq_km,
                        ST_GeometryType(boundary),
                        ST_SRID(boundary)
                    FROM plots
                    WHERE project_id = %s
                    ORDER BY plot_id
                    """,
                    (project_id,)
                )
                rows = cursor.fetchall()
        finally:
            connection.close()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Plot retrieval failed: {exc}"
        )
    return {
        "project_id": project_id,
        "count": len(rows),
        "plots": [
            {
                "plot_id": row[0],
                "project_id": row[1],
                "plot_name": row[2],
                "area_sq_km": float(row[3]),
                "geometry_type": row[4],
                "srid": row[5],
            }
            for row in rows
        ]
    }
@app.get("/api/plots/{plot_id}")
def get_plot(plot_id: int):
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        plot_id,
                        project_id,
                        plot_name,
                        area_sq_km,
                        ST_AsGeoJSON(boundary),
                        ST_GeometryType(boundary),
                        ST_SRID(boundary)
                    FROM plots
                    WHERE plot_id = %s
                    """,
                    (plot_id,)
                )
                row = cursor.fetchone()
        finally:
            connection.close()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Plot retrieval failed: {exc}"
        )
    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Plot not found."
        )
    import json
    return {
        "status": "success",
        "plot": {
            "plot_id": row[0],
            "project_id": row[1],
            "plot_name": row[2],
            "area_sq_km": float(row[3]),
            "geometry": json.loads(row[4]),
            "geometry_type": row[5],
            "srid": row[6],
        }
    }
# ============================================================
# SOIL
# ============================================================
@app.get("/api/plots/{plot_id}/soil")
def get_plot_soil(plot_id: int):
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        soil_id,
                        plot_id,
                        soil_type,
                        ph,
                        organic_matter,
                        moisture_percentage,
                        permeability,
                        permeability_unit,
                        bearing_capacity,
                        bearing_capacity_unit,
                        data_source,
                        source_reference,
                        data_confidence,
                        notes,
                        analyzed_at
                    FROM soil_analysis
                    WHERE plot_id = %s
                    ORDER BY analyzed_at DESC NULLS LAST
                    """,
                    (plot_id,)
                )
                rows = cursor.fetchall()
        finally:
            connection.close()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Soil retrieval failed: {exc}"
        )
    records = []
    for row in rows:
        records.append(
            {
                "soil_id": row[0],
                "plot_id": row[1],
                "soil_type": row[2],
                "ph": row[3],
                "organic_matter": row[4],
                "moisture_percentage": row[5],
                "permeability": row[6],
                "permeability_unit": row[7],
                "bearing_capacity": row[8],
                "bearing_capacity_unit": row[9],
                "data_source": row[10],
                "source_reference": row[11],
                "data_confidence": row[12],
                "notes": row[13],
                "analyzed_at": json_value(row[14]),
            }
        )
    return {
        "plot_id": plot_id,
        "count": len(records),
        "soil_records": records,
    }
@app.post("/api/plots/{plot_id}/soil")
def create_soil_record(
    plot_id: int,
    payload: SoilAnalysisCreate
):
    try:
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT 1
                    FROM plots
                    WHERE plot_id = %s
                    """,
                    (plot_id,)
                )
                if cursor.fetchone() is None:
                    raise HTTPException(
                        status_code=404,
                        detail="Plot not found."
                    )
                cursor.execute(
                    """
                    INSERT INTO soil_analysis
                    (
                        plot_id,
                        soil_type,
                        ph,
                        organic_matter,
                        moisture_percentage,
                        permeability,
                        permeability_unit,
                        bearing_capacity,
                        bearing_capacity_unit,
                        data_source,
                        source_reference,
                        data_confidence,
                        notes,
                        analyzed_at
                    )
                    VALUES
                    (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        NOW()
                    )
                    RETURNING soil_id
                    """,
                    (
                        plot_id,
                        payload.soil_type,
                        payload.ph,
                        payload.organic_matter,
                        payload.moisture_percentage,
                        payload.permeability,
                        payload.permeability_unit,
                        payload.bearing_capacity,
                        payload.bearing_capacity_unit,
                        payload.data_source,
                        payload.source_reference,
                        payload.data_confidence,
                        payload.notes,
                    )
                )
                soil_id = cursor.fetchone()[0]
            connection.commit()
        finally:
            connection.close()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Soil record creation failed: {exc}"
        )
    return {
        "status": "success",
        "soil_id": soil_id,
        "plot_id": plot_id,
        "message": "Soil evidence stored successfully."
    }
# ============================================================
# FRONTEND
# ============================================================
# ============================================================
# NEXORA SITE INTELLIGENCE
# ============================================================
@app.get("/api/site/intelligence")
def site_intelligence(
    latitude: float = Query(...),
    longitude: float = Query(...),
    plot_id: Optional[int] = Query(default=None),
    radius_km: float = Query(
        10.0,
        ge=1.0,
        le=500.0
    ),
):
    """
    Unified NEXORA site intelligence endpoint.
    Combines:
    - Location
    - Climate
    - Historical flood evidence
    - Flood ML
    - Plot/PostGIS information
    - Soil evidence
    Important:
    Missing evidence is reported as missing.
    NEXORA does not invent engineering or environmental values.
    """
    validate_coordinates(latitude, longitude)
    result = {
        "status": "success",
        "nexora": {
            "platform": "NEXORA",
            "module": "Site Intelligence",
            "version": app.version,
        },
        "location": {
            "latitude": latitude,
            "longitude": longitude,
        },
        "plot": None,
        "climate": None,
        "historical_flood": None,
        "flood_ml": None,
        "soil": None,
        "decision_intelligence": {
            "status": "DATA LIMITED",
            "confidence": "LOW",
            "evidence_count": 0,
            "available_modules": [],
            "missing_modules": [],
        },
        "provenance": [],
    }
    # ========================================================
    # 1. CLIMATE
    # ========================================================
    try:
        climate_result = analyze_climate(
            latitude=latitude,
            longitude=longitude
        )
        result["climate"] = climate_result
        result["decision_intelligence"][
            "available_modules"
        ].append("CLIMATE")
        result["decision_intelligence"][
            "evidence_count"
        ] += 1
        result["provenance"].append(
            {
                "module": "Climate",
                "source": "Open-Meteo Historical Weather API",
                "status": "AVAILABLE",
            }
        )
    except Exception as exc:
        result["decision_intelligence"][
            "missing_modules"
        ].append("CLIMATE")
        result["provenance"].append(
            {
                "module": "Climate",
                "status": "UNAVAILABLE",
                "reason": str(exc),
            }
        )
    # ========================================================
    # 2. HISTORICAL FLOOD
    # ========================================================
    try:
        flood_result = analyze_flood(
            latitude=latitude,
            longitude=longitude,
            radius_km=radius_km
        )
        result["historical_flood"] = flood_result
        result["decision_intelligence"][
            "available_modules"
        ].append("HISTORICAL_FLOOD")
        result["decision_intelligence"][
            "evidence_count"
        ] += 1
        result["provenance"].append(
            {
                "module": "Historical Flood",
                "source": "Global Flood Database",
                "radius_km": radius_km,
                "status": "AVAILABLE",
            }
        )
    except Exception as exc:
        result["decision_intelligence"][
            "missing_modules"
        ].append("HISTORICAL_FLOOD")
        result["provenance"].append(
            {
                "module": "Historical Flood",
                "status": "UNAVAILABLE",
                "reason": str(exc),
            }
        )
    # ========================================================
    # 3. FLOOD ML
    # ========================================================
    try:
        ml_result = analyze_flood_ml(
            latitude=latitude,
            longitude=longitude
        )
        result["flood_ml"] = ml_result
        result["decision_intelligence"][
            "available_modules"
        ].append("FLOOD_ML")
        result["decision_intelligence"][
            "evidence_count"
        ] += 1
        result["provenance"].append(
            {
                "module": "Flood ML",
                "source": "NEXORA calibrated flood ML model",
                "status": "AVAILABLE",
            }
        )
    except Exception as exc:
        result["decision_intelligence"][
            "missing_modules"
        ].append("FLOOD_ML")
        result["provenance"].append(
            {
                "module": "Flood ML",
                "status": "UNAVAILABLE",
                "reason": str(exc),
            }
        )
    # ========================================================
    # 4. PLOT / POSTGIS
    # ========================================================
    if plot_id is not None:
        try:
            connection = get_db_connection()
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT
                            plot_id,
                            project_id,
                            plot_name,
                            area_sq_km,
                            ST_AsGeoJSON(boundary),
                            ST_GeometryType(boundary),
                            ST_SRID(boundary)
                        FROM plots
                        WHERE plot_id = %s
                        """,
                        (plot_id,)
                    )
                    row = cursor.fetchone()
            finally:
                connection.close()
            if row is not None:
                import json
                result["plot"] = {
                    "plot_id": row[0],
                    "project_id": row[1],
                    "plot_name": row[2],
                    "area_sq_km": float(row[3]),
                    "geometry": json.loads(row[4]),
                    "geometry_type": row[5],
                    "srid": row[6],
                }
                result["decision_intelligence"][
                    "available_modules"
                ].append("PLOT_GEOMETRY")
                result["decision_intelligence"][
                    "evidence_count"
                ] += 1
                result["provenance"].append(
                    {
                        "module": "Plot Geometry",
                        "source": "NEXORA PostgreSQL/PostGIS",
                        "plot_id": plot_id,
                        "status": "AVAILABLE",
                    }
                )
            else:
                result["decision_intelligence"][
                    "missing_modules"
                ].append("PLOT_GEOMETRY")
        except Exception as exc:
            result["decision_intelligence"][
                "missing_modules"
            ].append("PLOT_GEOMETRY")
            result["provenance"].append(
                {
                    "module": "Plot Geometry",
                    "status": "UNAVAILABLE",
                    "reason": str(exc),
                }
            )
    else:
        result["decision_intelligence"][
            "missing_modules"
        ].append("PLOT_GEOMETRY")
    # ========================================================
    # 5. SOIL
    # ========================================================
    if plot_id is not None:
        try:
            connection = get_db_connection()
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT
                            soil_id,
                            plot_id,
                            soil_type,
                            ph,
                            organic_matter,
                            moisture_percentage,
                            permeability,
                            permeability_unit,
                            bearing_capacity,
                            bearing_capacity_unit,
                            data_source,
                            source_reference,
                            data_confidence,
                            notes,
                            analyzed_at
                        FROM soil_analysis
                        WHERE plot_id = %s
                        ORDER BY analyzed_at DESC NULLS LAST
                        LIMIT 1
                        """,
                        (plot_id,)
                    )
                    row = cursor.fetchone()
            finally:
                connection.close()
            if row is not None:
                result["soil"] = {
                    "soil_id": row[0],
                    "plot_id": row[1],
                    "soil_type": row[2],
                    "ph": row[3],
                    "organic_matter": row[4],
                    "moisture_percentage": row[5],
                    "permeability": row[6],
                    "permeability_unit": row[7],
                    "bearing_capacity": row[8],
                    "bearing_capacity_unit": row[9],
                    "data_source": row[10],
                    "source_reference": row[11],
                    "data_confidence": row[12],
                    "notes": row[13],
                    "analyzed_at": json_value(row[14]),
                }
                result["decision_intelligence"][
                    "available_modules"
                ].append("SOIL")
                result["decision_intelligence"][
                    "evidence_count"
                ] += 1
                result["provenance"].append(
                    {
                        "module": "Soil",
                        "source": row[10],
                        "source_reference": row[11],
                        "confidence": row[12],
                        "status": "AVAILABLE",
                    }
                )
            else:
                result["decision_intelligence"][
                    "missing_modules"
                ].append("SOIL")
        except Exception as exc:
            result["decision_intelligence"][
                "missing_modules"
            ].append("SOIL")
            result["provenance"].append(
                {
                    "module": "Soil",
                    "status": "UNAVAILABLE",
                    "reason": str(exc),
                }
            )
    else:
        result["decision_intelligence"][
            "missing_modules"
        ].append("SOIL")
    # ========================================================
    # 6. EVIDENCE CONFIDENCE
    # ========================================================
    evidence_count = result["decision_intelligence"][
        "evidence_count"
    ]
    missing_count = len(
        result["decision_intelligence"]["missing_modules"]
    )
    if evidence_count >= 5 and missing_count == 0:
        confidence = "HIGH"
    elif evidence_count >= 3:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    result["decision_intelligence"]["confidence"] = confidence
    # ========================================================
    # 7. DECISION POSTURE
    # ========================================================
    ml_probability = None
    try:
        ml_prediction = result["flood_ml"]["prediction"]
        ml_probability = float(
            ml_prediction["flood_probability"]
        )
    except Exception:
        try:
            ml_prediction = result["flood_ml"][
                "ml_prediction"
            ]
            ml_probability = float(
                ml_prediction["flood_probability"]
            )
        except Exception:
            ml_probability = None
    historical_events = 0
    try:
        historical_events = int(
            result["historical_flood"]["event_count"]
        )
    except Exception:
        historical_events = 0
    if ml_probability is None:
        posture = "INSUFFICIENT EVIDENCE"
    elif ml_probability >= 0.80:
        posture = "HIGH RISK SIGNAL"
    elif ml_probability >= 0.60:
        posture = "ELEVATED RISK SIGNAL"
    elif ml_probability >= 0.40:
        posture = "MODERATE RISK SIGNAL"
    else:
        posture = "LOWER RISK SIGNAL"
    result["decision_intelligence"]["decision_posture"] = posture
    result["decision_intelligence"]["flood_ml_probability"] = (
        round(ml_probability * 100, 2)
        if ml_probability is not None
        else None
    )
    result["decision_intelligence"]["historical_event_count"] = (
        historical_events
    )
    result["decision_intelligence"]["interpretation"] = (
        "NEXORA provides evidence-based decision intelligence. "
        "It does not certify a site as safe or unsafe. "
        "Missing evidence lowers confidence."
    )
    return result
# ============================================================
# TERRAIN INTELLIGENCE
# ============================================================
@app.get("/api/analyze/terrain")
def terrain_analysis(
    latitude: float = Query(...),
    longitude: float = Query(...),
):
    validate_coordinates(latitude, longitude)
    try:
        return analyze_terrain(
            latitude=latitude,
            longitude=longitude,
        )
    except  Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Terrain data source unavailable: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Terrain analysis failed: {exc}",
        ) from exc
@app.get("/api/analyze/infrastructure")
def infrastructure_analysis(
    latitude: float = Query(...),
    longitude: float = Query(...),
    radius_km: float = Query(25, ge=1, le=100),
):
    validate_coordinates(latitude, longitude)
    try:
        return analyze_infrastructure(
            latitude=latitude,
            longitude=longitude,
            radius_km=radius_km,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Infrastructure data source unavailable: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Infrastructure analysis failed: {exc}",
        ) from exc
@app.get("/api/analyze/route")
def route_analysis(
    origin_latitude: float = Query(...),
    origin_longitude: float = Query(...),
    destination_latitude: float = Query(...),
    destination_longitude: float = Query(...),
):
    try:
        result = get_route(
            origin_latitude=origin_latitude,
            origin_longitude=origin_longitude,
            destination_latitude=destination_latitude,
            destination_longitude=destination_longitude,
        )
        return {
            "status": "success",
            "analysis": result,
            "source": "OSRM Routing API",
        }
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Route data source unavailable: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Route analysis failed: {exc}",
        ) from exc
@app.get("/api/analyze/scenarios")
def get_scenario_intelligence(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(10, gt=0, le=500),
):
    validate_coordinates(latitude, longitude)
    end_date = date.today() - timedelta(days=7)
    start_date = end_date - timedelta(days=365)

    try:
        climate_payload = extract_climate_data(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        climate_frame = climate_dataframe(climate_payload)
        if climate_frame.empty:
            raise HTTPException(
                status_code=422,
                detail="No climate observations available for scenario analysis.",
            )

        climate = {
            "status": "success",
            "location": {"latitude": latitude, "longitude": longitude},
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
            "observation_count": int(len(climate_frame)),
            "climate": {
                "temperature_mean": float(climate_frame["temperature_mean"].mean()),
                "temperature_max": float(climate_frame["temperature_max"].max()),
                "temperature_min": float(climate_frame["temperature_min"].min()),
                "precipitation_sum": float(climate_frame["precipitation"].sum()),
                "precipitation_mean": float(climate_frame["precipitation"].mean()),
                "precipitation_max": float(climate_frame["precipitation"].max()),
                "precipitation_p95": float(climate_frame["precipitation"].quantile(0.95)),
                "heavy_rain_days": int((climate_frame["precipitation"] >= 20).sum()),
                "extreme_rain_days": int((climate_frame["precipitation"] >= 50).sum()),
            },
            "source": "Open-Meteo Historical Weather API",
        }
        flood = analyze_flood(
            latitude=latitude,
            longitude=longitude,
            radius_km=radius_km,
        )
        return analyze_scenarios(climate=climate, flood=flood)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Scenario analysis failed: {type(exc).__name__}: {exc}",
        ) from exc


@app.post("/api/analyze/risk-response")
def risk_response_analysis(payload: dict[str, Any]):
    try:
        return build_risk_response(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Risk-response analysis failed: {exc}",
        ) from exc


@app.get("/api/analyze/route")
def route_analysis(
    origin_latitude: float,
    origin_longitude: float,
    destination_latitude: float,
    destination_longitude: float,
):
    try:
        return get_route(
            origin_latitude=origin_latitude,
            origin_longitude=origin_longitude,
            destination_latitude=destination_latitude,
            destination_longitude=destination_longitude,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Route analysis failed: {exc}",
        ) from exc


# NEXORA_ANALYSIS_PERSISTENCE_V1
@app.middleware("http")
async def nexora_persist_analysis_runs(request, call_next):
    """Persist successful analysis responses without changing their payload."""
    from starlette.responses import Response
    import json
    import asyncio
    import os

    tracked = (
        "/api/site/intelligence",
        "/api/analyze/flood",
        "/analyze/climate",
        "/api/analyze/terrain",
        "/api/analyze/infrastructure",
        "/api/analyze/route",
    )

    response = await call_next(request)
    if request.method != "GET" or request.url.path not in tracked:
        return response

    chunks = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        chunks.append(chunk)
    body = b"".join(chunks)

    try:
        payload = json.loads(body.decode("utf-8"))
        if 200 <= response.status_code < 300:
            import psycopg2
            conn = psycopg2.connect(
                dbname=os.getenv("PGDATABASE", "nexora_db"),
                host=os.getenv("PGHOST") or None,
                port=os.getenv("PGPORT") or None,
                user=os.getenv("PGUSER") or None,
                password=os.getenv("PGPASSWORD") or None,
            )
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO public.analysis_runs
                            (endpoint, request_params, response_payload, http_status)
                        VALUES (%s, %s::jsonb, %s::jsonb, %s)
                        """,
                        (
                            request.url.path,
                            json.dumps(dict(request.query_params)),
                            json.dumps(payload),
                            response.status_code,
                        ),
                    )
                conn.commit()
            finally:
                conn.close()
    except Exception:
        # Persistence failure must not turn a successful analysis into an API error.
        import logging
        logging.getLogger("nexora.persistence").exception(
            "Could not persist analysis response for %s", request.url.path
        )

    return Response(
        content=body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
        background=response.background,
    )


# NEXORA_ANALYSIS_HISTORY_V2
# Stores successful JSON analysis responses independently of domain tables.
import json as _nexora_json
import logging as _nexora_logging
import os as _nexora_os
import time as _nexora_time
from datetime import datetime as _nexora_datetime
from fastapi import Request as _NexoraRequest
from fastapi.responses import Response as _NexoraResponse

_NEXORA_TRACKED_PATHS = {
    "/api/site/intelligence",
    "/api/analyze/flood",
    "/analyze/climate",
    "/api/analyze/terrain",
    "/api/analyze/infrastructure",
    "/api/analyze/route",
    "/api/soil/geographic",
}

def _nexora_db_connect():
    import psycopg2
    return psycopg2.connect(
        dbname=_nexora_os.getenv("PGDATABASE", "nexora_db"),
        **({
            key: value for key, value in {
                "host": _nexora_os.getenv("PGHOST"),
                "port": _nexora_os.getenv("PGPORT"),
                "user": _nexora_os.getenv("PGUSER"),
                "password": _nexora_os.getenv("PGPASSWORD"),
            }.items() if value
        })
    )

def _nexora_ensure_history_table():
    conn = _nexora_db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS public.analysis_run_records (
                    id BIGSERIAL PRIMARY KEY,
                    endpoint TEXT NOT NULL,
                    http_method TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    query_params JSONB NOT NULL DEFAULT '{}'::jsonb,
                    response_json JSONB,
                    response_excerpt TEXT,
                    duration_ms DOUBLE PRECISION,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
        conn.commit()
    finally:
        conn.close()

try:
    _nexora_ensure_history_table()
except Exception:
    _nexora_logging.getLogger("nexora.persistence").exception(
        "Analysis history table initialization failed; API startup will continue"
    )

@app.middleware("http")
async def nexora_analysis_history_middleware(request: _NexoraRequest, call_next):
    if request.method != "GET" or request.url.path not in _NEXORA_TRACKED_PATHS:
        return await call_next(request)

    started = _nexora_time.perf_counter()
    response = await call_next(request)

    # Only capture JSON success responses; don't consume or rewrite other responses.
    content_type = response.headers.get("content-type", "")
    if response.status_code < 200 or response.status_code >= 300 or "json" not in content_type.lower():
        return response

    try:
        chunks = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                chunks.append(chunk)
            else:
                chunks.append(str(chunk).encode("utf-8"))
        body = b"".join(chunks)

        parsed = None
        try:
            parsed = _nexora_json.loads(body.decode("utf-8"))
        except Exception:
            pass

        # Rebuild the response so the frontend receives the original body.
        headers = dict(response.headers)
        headers.pop("content-length", None)
        rebuilt = _NexoraResponse(
            content=body,
            status_code=response.status_code,
            headers=headers,
            media_type=content_type.split(";")[0],
            background=response.background,
        )

        try:
            conn = _nexora_db_connect()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO public.analysis_run_records
                            (endpoint, http_method, status_code, query_params,
                             response_json, response_excerpt, duration_ms)
                        VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
                    """, (
                        request.url.path,
                        request.method,
                        response.status_code,
                        _nexora_json.dumps(dict(request.query_params)),
                        _nexora_json.dumps(parsed) if parsed is not None else None,
                        body[:4000].decode("utf-8", errors="replace"),
                        round((_nexora_time.perf_counter() - started) * 1000, 2),
                    ))
                conn.commit()
            finally:
                conn.close()
        except Exception:
            _nexora_logging.getLogger("nexora.persistence").exception(
                "Could not save analysis history for %s", request.url.path
            )

        return rebuilt
    except Exception:
        _nexora_logging.getLogger("nexora.persistence").exception(
            "Could not capture analysis response for %s", request.url.path
        )
        # The middleware must not turn an analysis response into a server error.
        return response

@app.get("/api/analysis/history")
def nexora_analysis_history(limit: int = 20, endpoint: str = None):
    """Return recent saved analysis runs; newest first."""
    limit = max(1, min(limit, 100))
    conn = _nexora_db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, endpoint, http_method, status_code, query_params,
                       response_json, response_excerpt, duration_ms, created_at
                FROM public.analysis_run_records
                WHERE (%s IS NULL OR endpoint = %s)
                ORDER BY created_at DESC
                LIMIT %s
            """, (endpoint, endpoint, limit))
            rows = cur.fetchall()
        return {
            "status": "success",
            "count": len(rows),
            "runs": [
                {
                    "id": row[0],
                    "endpoint": row[1],
                    "method": row[2],
                    "status_code": row[3],
                    "query_params": row[4],
                    "response": row[5],
                    "response_excerpt": row[6],
                    "duration_ms": row[7],
                    "created_at": row[8].isoformat() if row[8] else None,
                }
                for row in rows
            ],
            "note": "Saved API responses are evidence snapshots, not independent engineering validation."
        }
    finally:
        conn.close()



# NEXORA_HOSTING_FOUNDATION_V1
from fastapi.responses import FileResponse as _NexoraFileResponse
from fastapi import Request as _NexoraHostRequest
import secrets as _nexora_secrets
import base64 as _nexora_base64
import hmac as _nexora_hmac

@app.get("/", include_in_schema=False)
def nexora_landing_page():
    return _NexoraFileResponse(FRONTEND_DIR / "index.html")

@app.get("/index.html", include_in_schema=False)
def nexora_landing_page_file():
    return _NexoraFileResponse(FRONTEND_DIR / "index.html")

@app.get("/dashboard.html", include_in_schema=False)
def nexora_dashboard_page():
    return _NexoraFileResponse(FRONTEND_DIR / "dashboard.html")

@app.middleware("http")
async def nexora_host_access_control(request: _NexoraHostRequest, call_next):
    enabled = os.getenv("NEXORA_AUTH_ENABLED", "false").lower() == "true"
    if enabled and request.url.path != "/api/health":
        username = os.getenv("NEXORA_AUTH_USERNAME", "")
        password = os.getenv("NEXORA_AUTH_PASSWORD", "")
        if not username or not password:
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(
                "NEXORA access is not configured.", status_code=503
            )
        supplied = request.headers.get("authorization", "")
        valid = False
        if supplied.startswith("Basic "):
            try:
                decoded = _nexora_base64.b64decode(
                    supplied[6:], validate=True
                ).decode("utf-8")
                candidate_user, candidate_password = decoded.split(":", 1)
                valid = (
                    _nexora_secrets.compare_digest(candidate_user, username)
                    and _nexora_secrets.compare_digest(candidate_password, password)
                )
            except (ValueError, UnicodeError):
                valid = False
        if not valid:
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(
                "Authentication required.",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="NEXORA"'},
            )
    return await call_next(request)
