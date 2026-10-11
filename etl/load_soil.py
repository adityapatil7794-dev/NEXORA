import os
from datetime import datetime

import psycopg2

from etl.soil_intelligence import get_soil_dataset


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "nexora_db",
    "user": os.getenv("adityaanilpatil"),
    "password": os.getenv("aditya3971"),
}


def load_soil_analysis(plot_id: int, latitude: float, longitude: float):

    soil = get_soil_dataset(latitude, longitude)

    if soil["ph"] is None:
        raise ValueError("SoilGrids returned no valid pH value.")

    connection = psycopg2.connect(**DB_CONFIG)

    try:
        with connection:
            with connection.cursor() as cursor:

                cursor.execute(
                    """
                    INSERT INTO soil_analysis (
                        plot_id,
                        soil_type,
                        ph,
                        organic_matter,
                        moisture_percent,
                        permeability,
                        bearing_capacity,
                        data_source,
                        source_reference,
                        data_confidence,
                        notes,
                        analyzed_at,
                        clay_g_kg,
                        sand_g_kg,
                        silt_g_kg,
                        bulk_density_g_cm3,
                        cec_mmol_c_kg,
                        organic_carbon_dg_kg,
                        coarse_fragments_percent,
                        resolution_m,
                        depth_interval
                    )
                    VALUES (
                        %s,
                        NULL,
                        %s,
                        NULL,
                        NULL,
                        NULL,
                        NULL,
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
                        %s
                    )
                    RETURNING soil_id;
                    """,
                    (
                        plot_id,
                        soil["ph"]["value"],
                        "ISRIC SoilGrids",
                        "SoilGrids 250 m global dataset",
                        "MEDIUM",
                        "Modeled SoilGrids data; engineering properties require verified field/lab sources.",
                        datetime.utcnow(),

                        soil["clay"]["value"],
                        soil["sand"]["value"],
                        soil["silt"]["value"],
                        soil["bulk_density"]["value"],
                        soil["cec"]["value"],
                        soil["organic_carbon"]["value"],
                        soil["coarse_fragments"]["value"],

                        250,
                        "0-5 cm",
                    ),
                )

                soil_id = cursor.fetchone()[0]

        return soil_id

    finally:
        connection.close()


if __name__ == "__main__":

    soil_id = load_soil_analysis(
        plot_id=3,
        latitude=16.72335,
        longitude=74.54335,
    )

    print("=" * 60)
    print("NEXORA SOIL DATABASE LOAD")
    print("=" * 60)
    print(f"Soil ID: {soil_id}")
    print("Status: SUCCESS")