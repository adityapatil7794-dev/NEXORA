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