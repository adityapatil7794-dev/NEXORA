import subprocess
from pathlib import Path
from pyproj import Transformer
import rasterio


# ---------------------------------------------------------
# NEXORA Soil Intelligence
# Source: ISRIC SoilGrids
# Resolution: 250 m
# Depth: 0-5 cm
# ---------------------------------------------------------

BASE_URL = (
    "https://files.isric.org/soilgrids/latest/data"
)

DATA_DIR = Path("data/soil_cache")
DATA_DIR.mkdir(parents=True, exist_ok=True)

TRANSFORMER = Transformer.from_crs(
    "EPSG:4326",
    "ESRI:54052",
    always_xy=True
)


SOIL_PROPERTIES = {
    "ph": {
        "folder": "phh2o",
        "file": "phh2o_0-5cm_Q0.5.vrt",
        "scale": 10,
        "unit": "pH"
    },
    "clay": {
        "folder": "clay",
        "file": "clay_0-5cm_Q0.5.vrt",
        "scale": 10,
        "unit": "g/kg"
    },
    "sand": {
        "folder": "sand",
        "file": "sand_0-5cm_Q0.5.vrt",
        "scale": 10,
        "unit": "g/kg"
    },
    "silt": {
        "folder": "silt",
        "file": "silt_0-5cm_Q0.5.vrt",
        "scale": 10,
        "unit": "g/kg"
    },
    "bulk_density": {
        "folder": "bdod",
        "file": "bdod_0-5cm_Q0.5.vrt",
        "scale": 100,
        "unit": "g/cm³"
    },
    "cec": {
        "folder": "cec",
        "file": "cec_0-5cm_Q0.5.vrt",
        "scale": 10,
        "unit": "mmol(c)/kg"
    },
    "organic_carbon": {
        "folder": "soc",
        "file": "soc_0-5cm_Q0.5.vrt",
        "scale": 10,
        "unit": "dg/kg"
    },
    "coarse_fragments": {
        "folder": "cfvo",
        "file": "cfvo_0-5cm_Q0.5.vrt",
        "scale": 10,
        "unit": "%"
    }
}


def build_vrt_url(property_name):
    info = SOIL_PROPERTIES[property_name]

    return (
        "/vsicurl?max_retry=3"
        "&retry_delay=1"
        "&list_dir=no"
        f"&url={BASE_URL}/"
        f"{info['folder']}/"
        f"{info['file']}"
    )


def extract_property(latitude, longitude, property_name):
    """
    Extract a 1 km × 1 km SoilGrids window
    around the requested coordinate.
    """

    x, y = TRANSFORMER.transform(longitude, latitude)

    half_size = 500

    xmin = x - half_size
    xmax = x + half_size
    ymin = y - half_size
    ymax = y + half_size

    output = DATA_DIR / (
        f"{property_name}_"
        f"{latitude:.5f}_"
        f"{longitude:.5f}.tif"
    )

    if not output.exists():

        vrt = build_vrt_url(property_name)

        command = [
            "gdal_translate",

            "-projwin",
            str(xmin),
            str(ymax),
            str(xmax),
            str(ymin),

            "-projwin_srs",
            (
                "+proj=igh "
                "+lat_0=0 "
                "+lon_0=0 "
                "+datum=WGS84 "
                "+units=m "
                "+no_defs"
            ),

            "-tr",
            "250",
            "250",

            "-co",
            "TILED=YES",

            "-co",
            "COMPRESS=DEFLATE",

            "-co",
            "PREDICTOR=2",

            vrt,
            str(output)
        ]

        subprocess.run(
            command,
            check=True,
            timeout=180
        )

    with rasterio.open(output) as src:

        data = src.read(1).astype(float)

        nodata = src.nodata

        if nodata is not None:
            data[data == nodata] = float("nan")

        valid = data[~__import__("numpy").isnan(data)]

        if len(valid) == 0:
            return None

        raw_mean = float(valid.mean())

    scale = SOIL_PROPERTIES[property_name]["scale"]

    return {
        "value": raw_mean / scale,
        "unit": SOIL_PROPERTIES[property_name]["unit"],
        "resolution_m": 250,
        "source": "ISRIC SoilGrids",
        "depth": "0-5 cm"
    }


def get_soil_dataset(latitude, longitude):

    results = {}

    for property_name in SOIL_PROPERTIES:

        try:
            results[property_name] = extract_property(
                latitude,
                longitude,
                property_name
            )

        except Exception as exc:

            results[property_name] = {
                "value": None,
                "error": str(exc)
            }

    return results


if __name__ == "__main__":

    latitude = 16.72335
    longitude = 74.54335

    print("\nNEXORA SOIL INTELLIGENCE")
    print("=" * 50)

    result = get_soil_dataset(
        latitude,
        longitude
    )

    for name, value in result.items():

        print(f"\n{name}:")
        print(value)