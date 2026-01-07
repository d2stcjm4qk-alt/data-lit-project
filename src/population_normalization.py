import geopandas as gpd
from pathlib import Path
import os
import pandas as pd
import plotly.express as px


def load_accidents(
        path: str,
        category_filters: dict | None = None
):
    df = pd.read_csv(path, sep=",", dtype=str)

    df["longitude"] = (
        df["longitude"]
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    df["latitude"] = (
        df["latitude"]
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    # Apply user-specified filters
    if category_filters:
        for col, allowed in category_filters.items():
            df = df[df[col].isin(allowed)]

    return df


# Base directory (project root, one level up from src)
BASE_DIR = Path(__file__).resolve().parent.parent
regions = gpd.read_file(BASE_DIR / "data" / "processed" / "geo_data" / "Germany_merged.geojson")

regions.crs

germany_acc_path = BASE_DIR / "data" / "processed" / "reduced_uk_dataset" / "modified_ger.csv"
ger_acc_df = load_accidents(
    str(germany_acc_path),
    category_filters={
        "casualty_severity": ["1"],
    }
)

# Convert dataframe into a GeoDataFrame
accidents_gdf = gpd.GeoDataFrame(
    ger_acc_df,
    geometry=gpd.points_from_xy(
        ger_acc_df.longitude,
        ger_acc_df.latitude
    ),
    crs="EPSG:4326"
)

accidents_gdf = accidents_gdf.to_crs(regions.crs)

accidents_with_region = gpd.sjoin(
    accidents_gdf,
    regions,
    how="left",
    predicate="within"  # or "intersects"
)

accident_counts = (
    accidents_with_region
    .groupby("region_code")
    .size()
    .rename("accident_count")
    .reset_index()
)

regions = regions.merge(
    accident_counts,
    on="region_code",
    how="left"
)

regions["accident_count"] = regions["accident_count"].fillna(0)
regions["accidents_per_100k"] = (
        regions["accident_count"] / regions["population"] * 100_000
)

