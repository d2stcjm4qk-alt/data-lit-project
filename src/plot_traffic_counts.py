import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import contextily as ctx
import pyproj
from pathlib import Path
import geopandas as gpd
from typing import Optional, Dict, Tuple, List, Union
import numpy as np
from hierarchical_poisson_model import load_accidents, add_accident_counts_to_regions


def plot_vehicles_per_km(
        regions_gdf: gpd.GeoDataFrame,
        column_aadf: str = "AADF_region_weighted",
        column_length: str = "osm_total_length_km",
        title: str = "Vehicles per km by Region",
        label: str = "Vehicles per km",
        save_path: Optional[str] = None,
        prominent_cities: Optional[Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
        simplify_tolerance: Optional[float] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        scale_width_m: Optional[float] = None,  # reference width for matching scale
        reuse_zoom: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None  # (xlim, ylim)
):
    """
    Plot vehicles per km per region using a GeoDataFrame.
    Uses log color scale by default. Optionally mark prominent cities.
    Returns (scale_width_m, (xlim, ylim)) for reuse.
    """

    # -------------------
    # Compute vehicles per km
    # -------------------
    gdf_plot = regions_gdf.copy()

    # Avoid division by zero
    gdf_plot["vehicles_per_km"] = gdf_plot[column_aadf].fillna(0) / gdf_plot[column_length].replace(0, np.nan)

    # -------------------
    # Simplify geometries if requested
    # -------------------
    if simplify_tolerance is not None:
        gdf_plot["geometry"] = gdf_plot["geometry"].simplify(tolerance=simplify_tolerance)

    # Project to Web Mercator for plotting
    gdf_plot = gdf_plot.to_crs(epsg=3857)

    # -------------------
    # Color scale
    # -------------------
    if vmin is None:
        vmin = gdf_plot["vehicles_per_km"].replace(0, np.nan).min()
    if vmax is None:
        vmax = gdf_plot["vehicles_per_km"].max()

    print(f'min: {vmin}, max: {vmax}')
    norm = mcolors.LogNorm(vmin=max(vmin, 1e-2), vmax=vmax)  # small epsilon to avoid log(0)

    # -------------------
    # Plot
    # -------------------
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16
    })

    fig, ax = plt.subplots(figsize=(12, 14))
    gdf_plot.plot(
        column="vehicles_per_km",
        cmap="viridis",
        legend=True,
        ax=ax,
        edgecolor="gray",
        linewidth=0.3,
        alpha=0.8,
        norm=norm,
        legend_kwds={'label': label, 'shrink': 0.6}
    )

    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=7, alpha=0.5)

    # -------------------
    # Physical scale / zoom
    # -------------------
    if reuse_zoom is not None:
        xlim, ylim = reuse_zoom
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        scale_width_m = xlim[1] - xlim[0]
    else:
        minx, miny, maxx, maxy = gdf_plot.total_bounds
        real_width = maxx - minx
        real_height = maxy - miny
        aspect = real_height / real_width

        if scale_width_m is None:
            scale_width_m = real_width

        scale_width_m = max(scale_width_m, real_width)
        target_height_m = max(scale_width_m * aspect, real_height)

        cx = (minx + maxx) / 2
        cy = (miny + maxy) / 2

        xlim = (cx - scale_width_m / 2, cx + scale_width_m / 2)
        ylim = (cy - target_height_m / 2, cy + target_height_m / 2)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # -------------------
    # Plot prominent cities
    # -------------------
    if prominent_cities:
        project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        for label, ((lon, lat), (dx, dy)) in prominent_cities.items():
            x, y = project.transform(lon, lat)
            ax.plot(x, y, "o", color='0.3', markersize=4)
            ax.text(
                x + dx, y + dy,
                rf"\textbf{{{label}}}",
                fontsize=14,
                ha="center",
                va="bottom",
                color='0.3'
            )

    ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return scale_width_m, (xlim, ylim)


def plot_interactive_plotly(gdf, value_col="population", name_col="region_code"):
    gdf = gdf.to_crs(4326)

    fig = px.choropleth(
        gdf,
        geojson=gdf.geometry,
        locations=gdf.index,
        color=value_col,
        hover_data=[name_col, value_col],
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.show()


def main():
    # Base directory (project root, one level up from src)
    BASE_DIR = Path(__file__).resolve().parent.parent

    # germany_geo_path = BASE_DIR / "data" / "processed" / "geo_data" / "Germany_merged.geojson"
    germany_geo_path = BASE_DIR / "data" / "preprocessed" / "germany" / "traffic" / "ger_gdf_with_osm_roads.gpkg"
    ger_regions_gdf = gpd.read_file(germany_geo_path)
    print(ger_regions_gdf.columns)
    plot_vehicles_per_km(ger_regions_gdf, vmax=1045,
                         save_path=BASE_DIR / "results" / "figures" / "germany" / "germany_traffic_volume_per_region.png")

    germany_acc_path = BASE_DIR / "data" / "processed" / "reduced_uk_dataset" / "modified_ger.csv"

    ger_accidents_gdf = load_accidents(
        str(germany_acc_path),
        category_filters={
            "casualty_severity": [1],
            "is_mcyle": [1]
        }
    )
    ger_regions_with_accidents = add_accident_counts_to_regions(
        regions_gdf=ger_regions_gdf,
        accidents_gdf=ger_accidents_gdf
    )
    ger_regions_with_accidents["vehicle_km_per_day"] = ger_regions_gdf["AADF_region_weighted"] / ger_regions_gdf[
        "osm_total_length_km"].replace(0, np.nan)
    ger_regions_with_accidents["accidents_per_vehicle_km"] = (
            ger_regions_with_accidents["accident_count"] / ger_regions_with_accidents["vehicle_km_per_day"].replace(0, np.nan)
    )
    plot_vehicles_per_km(ger_regions_with_accidents, column_aadf="accident_count", column_length="vehicle_km_per_day",
                         save_path=BASE_DIR / "results" / "figures" / "germany" / "germany_traffic_volume_per_region.png",
                         label="Accidents / Vehicles per km")

    uk_geo_path = BASE_DIR / "data" / "preprocessed" / "uk" / "traffic" / "uk_gdf_with_osm_roads.gpkg"
    uk_regions_gdf = gpd.read_file(uk_geo_path)
    plot_vehicles_per_km(uk_regions_gdf, vmax=1045,
                         save_path=BASE_DIR / "results" / "figures" / "uk" / "uk_traffic_volume_per_region.png")


if __name__ == "__main__":
    main()
