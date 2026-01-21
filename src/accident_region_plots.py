import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pyproj

from typing import Optional, Dict, Tuple
from pathlib import Path


def plot_accidents(
        regions_gdf: gpd.GeoDataFrame,
        accidents_gdf: gpd.GeoDataFrame,
        normalize_by_population: bool = False,
        title: str = "Traffic Accidents",
        save_path: Optional[str] = None,
        prominent_cities: Optional[Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
        simplify_tolerance: Optional[float] = None
):
    """
    Plot accidents per region using a cleaned + filtered accidents GeoDataFrame.

    Args:
        regions_gdf: GeoDataFrame with geometries + region_code (+ population optional).
        accidents_gdf: GeoDataFrame of filtered accident points in EPSG:4326.
        normalize_by_population: If True, plot accidents per 100k residents.
        title: Plot title.
        save_path: Path to save the output figure.
        prominent_cities: Dict of city name -> ((lon,lat),(dx,dy)) label offset.
        simplify_tolerance: Geometry simplification tolerance.
    """

    gdf_plot = regions_gdf.copy()

    # --- Simplify polygons if requested ---
    if simplify_tolerance:
        gdf_plot['geometry'] = gdf_plot['geometry'].simplify(simplify_tolerance)

    # --- Project regions to Web Mercator ---
    gdf_plot = gdf_plot.to_crs(epsg=3857)

    # Project accident points
    acc = accidents_gdf.to_crs(epsg=3857)

    # --- Count accidents by region polygon ---
    joined = gpd.sjoin(acc, gdf_plot, how="left", predicate="within")

    accident_counts = joined.groupby("index_right").size()
    gdf_plot["accident_count"] = accident_counts.reindex(gdf_plot.index).fillna(0).astype(int)

    # --- Normalization ---
    if normalize_by_population:
        if "population" not in gdf_plot.columns:
            raise ValueError("Missing 'population' column required for normalization.")
        gdf_plot["accidents_per_100k"] = (
                gdf_plot["accident_count"] / gdf_plot["population"] * 100000
        )

    column = "accidents_per_100k" if normalize_by_population else "accident_count"
    legend_label = "Accidents per 100k" if normalize_by_population else "Total accidents"

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 14))
    gdf_plot.plot(
        column=column,
        cmap="Reds",
        legend=True,
        legend_kwds={"label": legend_label},
        edgecolor="gray",
        linewidth=0.3,
        alpha=0.3,
        ax=ax
    )

    # Add basemap
    # ctx.add_basemap(ax, source=ctx.providers.Esri.WorldHillshade, alpha=1.0)

    # Hillshade overlay for terrain relief
    # ctx.add_basemap(ax, source=ctx.providers.Esri.WorldHillshade, alpha=0.35)

    # Add city markers
    if prominent_cities:
        project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        for label, ((lon, lat), (dx, dy)) in prominent_cities.items():
            x, y = project.transform(lon, lat)
            ax.plot(x, y, "o", color="0.3", markersize=2)
            ax.text(x + dx, y + dy, label, fontsize=6, fontweight="bold",
                    ha="center", va="bottom", color="0.3")

    ax.set_title(title)
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def load_accidents(
        path: str,
        category_filters: dict | None = None
) -> gpd.GeoDataFrame:
    """
    Load and clean the accidents dataset and return a filtered GeoDataFrame.

    Args:
        path: Path to the accident CSV file.
        category_filters: Dict of column -> list of allowed values.
                          Example: {"UKATEGORIE": [1, 2], "ULICHTVERH": [2]}

    Returns:
        A GeoDataFrame of accidents in EPSG:4326.
    """

    df = pd.read_csv(path, sep=";", dtype=str)

    # --- Normalize numeric columns (comma-decimal + scientific notation) ---
    for col in ["XGCSWGS84", "YGCSWGS84"]:
        df[col] = (
            df[col]
            .str.replace(",", ".", regex=False)  # German decimals
            .astype(float)
        )

    # Convert filterable columns back to numeric
    numeric_cols = [
        "UJAHR", "UMONAT", "USTUNDE", "UWOCHENTAG",
        "UKATEGORIE", "UART", "UTYP1", "ULICHTVERH",
        "IstPKW", "IstFuss", "IstRad", "IstKrad"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    # --- Apply user-specified filters ---
    if category_filters:
        for col, allowed in category_filters.items():
            df = df[df[col].isin(allowed)]

    # --- Create GeoDataFrame ---
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["XGCSWGS84"], df["YGCSWGS84"]),
        crs="EPSG:4326"
    )

    return gdf


def load_datasets(collision_path, vehicle_path):
    df_coll = pd.read_csv(collision_path, low_memory=False)
    df_veh = pd.read_csv(vehicle_path, low_memory=False)
    return df_coll, df_veh


# Mapping for UK -> German categories ---
def map_german_categories(vtype):
    # Bicycles
    if vtype == 1:
        return "Rad"  # IstRad

    # Motorcycles
    if vtype in [2, 3, 4, 5, 23, 97, 103, 104, 105, 106]:
        return "Krad"  # IstKrad

    # Passenger cars
    if vtype in [8, 9, 109]:
        return "PKW"  # IstPKW

    # Goods vehicles
    if vtype in [19, 20, 21, 98, 113]:
        return "Gkfz"  # IstGkfz

    # Everything else = Sonstige
    return "Sonstige"


def apply_vehicle_mapping(df_veh):
    df_veh["German_Class"] = df_veh["vehicle_type"].apply(map_german_categories)
    return df_veh


def create_indicators(df_veh):
    df_veh["IstRad"] = (df_veh["German_Class"] == "Rad").astype(int)
    df_veh["IstPKW"] = (df_veh["German_Class"] == "PKW").astype(int)
    df_veh["IstKrad"] = (df_veh["German_Class"] == "Krad").astype(int)
    df_veh["IstGkfz"] = (df_veh["German_Class"] == "Gkfz").astype(int)
    df_veh["IstSonstige"] = (df_veh["German_Class"] == "Sonstige").astype(int)

    # UK pedestrian info is not in vehicle table → always -1
    df_veh["IstFuss"] = -1

    return df_veh


def aggregate_collision_level(df_veh):
    df_summary = df_veh.groupby("collision_index").agg({
        "IstRad": "max",
        "IstPKW": "max",
        "IstKrad": "max",
        "IstGkfz": "max",
        "IstSonstige": "max",
        "IstFuss": "max"
    }).reset_index()

    return df_summary


def merge_with_collisions(df_coll, df_summary):
    df_final = df_coll.merge(df_summary, on="collision_index", how="left")
    df_final = df_final.fillna(0)  # if missing → set to 0
    return df_final


def save_dataset(df, output_path):
    df.to_csv(output_path, index=False, encoding="utf-8")


def build_german_aligned_uk_dataset(collision_path, vehicle_path, output_path=None):
    df_coll, df_veh = load_datasets(collision_path, vehicle_path)
    df_veh = apply_vehicle_mapping(df_veh)
    df_veh = create_indicators(df_veh)
    df_summary = aggregate_collision_level(df_veh)
    df_final = merge_with_collisions(df_coll, df_summary)

    # Save if path provided
    if output_path is not None:
        save_dataset(df_final, output_path)

    return df_final


def main():
    # Base directory (project root, one level up from src)
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Paths to data
    germany_acc_path = BASE_DIR / "data" / "raw" / "Germany" / "Unfallorte2024_LinRef.csv"
    germany_geo_path = BASE_DIR / "data" / "processed" / "geo_data" / "Germany_merged.geojson"
    kreise = gpd.read_file(germany_geo_path)

    accidents = load_accidents(
        germany_acc_path,
        category_filters={
            "UKATEGORIE": [1],  # deadly
            "IstKrad": [1]  # only darkness
        }
    )

    plot_accidents(
        regions_gdf=kreise,
        accidents_gdf=accidents,
        normalize_by_population=True,
        title="Filtered Traffic Accidents (Krad + category 1–2)"
    )


if __name__ == "__main__":
    main()
