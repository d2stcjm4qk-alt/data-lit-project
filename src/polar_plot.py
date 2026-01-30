import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import rcParams
from tueplots import bundles
import matplotlib.patches as mpatches

# Disable GeoJSON size limit for large files
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"  


SEASONS = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10, 11],
}


SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]

# Load datasets
def load_data():
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_dir_ger = BASE_DIR / "data" / "preprocessed" / "germany" / "collisions"/ "preprocessed_ger.csv"
    data_dir_uk = BASE_DIR / "data" / "preprocessed" / "uk" / "collisions"/ "preprocessed_uk.csv"

    df_de = pd.read_csv(
        data_dir_ger,
        usecols=["hour", "month", "casualty_severity", "longitude", "latitude"],
        low_memory=False,
    )

    df_uk = pd.read_csv(
        data_dir_uk,
        usecols=["time", "month", "casualty_severity", "longitude", "latitude"],
        low_memory=False,
    )

    return df_de, df_uk

def preprocess(df_de, df_uk):
    df_de = df_de[df_de["casualty_severity"] == 1].copy()
    df_uk = df_uk[df_uk["casualty_severity"] == 1].copy()

    # Germany: Filter valid hours
    df_de = df_de[(df_de["hour"] >= 0) & (df_de["hour"] <= 23)]
    df_de["hour"] = df_de["hour"].astype(int)

    # UK: Extract hour from timestamp
    df_uk = df_uk.dropna(subset=["time"]).copy()
    df_uk["hour"] = pd.to_numeric(df_uk["time"].astype(str).str[:2], errors="coerce")
    df_uk = df_uk.dropna(subset=["hour"])
    df_uk["hour"] = df_uk["hour"].astype(int)

    # Assign seasons based on month
    def month_to_season(month):
        for season, months in SEASONS.items():
            if month in months:
                return season
    df_de["season"] = df_de["month"].apply(month_to_season)
    df_uk["season"] = df_uk["month"].apply(month_to_season)

    return df_de, df_uk

# Calculate normalized hourly accident rates
def normalize_hourly(df_acc, regions_gdf, region_col="region_code"):
    # Standardize coordinate format
    for col in ["longitude", "latitude"]:
        df_acc[col] = df_acc[col].astype(str).str.replace(",", ".").astype(float)

    # Convert to GeoDataFrame
    acc_gdf = gpd.GeoDataFrame(
        df_acc,
        geometry=gpd.points_from_xy(df_acc.longitude, df_acc.latitude),
        crs="EPSG:4326"
    ).to_crs(regions_gdf.crs)

    # Spatial join to map accidents to regions
    joined = gpd.sjoin(acc_gdf, regions_gdf, how="left", predicate="within")

    # Compute hourly rate per 100k inhabitants
    hourly_df = (
        joined.groupby([region_col, "hour"])
        .agg({"population": "first", "geometry": "count"})
        .rename(columns={"geometry": "accident_count"})
        .reset_index()
    )
    hourly_df["accidents_per_100k"] = hourly_df["accident_count"] / hourly_df["population"] * 100_000

    # Calculate average rate across all regions, sum rates per hour and compute mean
    total_regions_count = len(regions_gdf)
    hourly_sum = hourly_df.groupby("hour")["accidents_per_100k"].sum()
    hourly_mean = hourly_sum / total_regions_count

    return hourly_mean.reindex(range(24), fill_value=0)


# plotting
def plot_hourly_clock(df_de, df_uk, regions_de_gdf, regions_uk_gdf):
    # tueplots settings
    bundle = bundles.icml2024(column="full", nrows=1, ncols=4)
    plt.rcParams.update(bundle)

    # Manual adjustments for figure dimensions
    current_width = plt.rcParams['figure.figsize'][0]
    plt.rcParams['figure.figsize'] = (current_width, 2.5) 
    plt.rcParams['figure.dpi'] = 200 
    base_fontsize = plt.rcParams['font.size']
    label_fontsize = base_fontsize + 3


    # Create figure and disable constrained_layout to allow manual spacing adjustments
    fig, axes = plt.subplots(
        1, 4,
        subplot_kw={"projection": "polar"},
        constrained_layout=False 
    )
    
    fig.set_size_inches(plt.rcParams['figure.figsize'][0], 3.5)

    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    width = 2 * np.pi / 28
    
    color_de = "darkorange"
    color_uk = "royalblue"
    max_val = 0
    plot_data = {}

    # prepare data for each season
    for season in SEASON_ORDER:
        df_de_season = df_de[df_de["season"] == season]
        df_uk_season = df_uk[df_uk["season"] == season]
        r_de = normalize_hourly(df_de_season, regions_de_gdf)
        r_uk = normalize_hourly(df_uk_season, regions_uk_gdf)
        plot_data[season] = (r_de, r_uk)
        max_val = max(max_val, r_de.max(), r_uk.max())

    # plot each season
    for i, season in enumerate(SEASON_ORDER):
        ax = axes[i]
        r_de, r_uk = plot_data[season]

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        ax.bar(angles, r_de.values, width=width, color=color_de, alpha=0.6)
        ax.bar(angles, r_uk.values, width=width * 0.75, color=color_uk, alpha=0.8)

        ax.set_title(season, pad=20, fontsize=label_fontsize+3, fontweight='bold')
        ax.set_xticks(angles)
        ax.set_xticklabels([str(h) if h % 2 == 0 else "" for h in range(24)], fontsize=base_fontsize +2.5)
        ax.tick_params(axis='x', pad=-4) 

        ax.set_ylim(0, max_val * 1.1)
        ref_vals = [max_val * 0.5, max_val]
        ax.set_yticks(ref_vals)
        ax.grid(True, alpha=0.4, color="gray", linewidth=0.7, linestyle="--")
        ax.set_yticklabels([]) 

        ax.set_yticklabels(
            [f"{v:.02f}" for v in ref_vals],
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=base_fontsize+2,
            fontweight='bold'
        )
    
    # layout adjustments
    plt.subplots_adjust(bottom=0.15, top=0.85, wspace=0.3, left=0.05, right=0.95)
    patch_de = mpatches.Patch(color=color_de, label='Germany', alpha=0.6)
    patch_uk = mpatches.Patch(color=color_uk, label='UK', alpha=0.8)

    # custom legend
    leg = fig.legend(
        handles=[patch_de, patch_uk],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.15), 
        ncol=2,
        frameon=False,
        fontsize=base_fontsize +4  
    )
    
    # Resize legend handles to match font size
    for handle in leg.legend_handles:
        handle.set_height(5)  
        handle.set_width(10)  

    plot_path= BASE_DIR / 'results' /"figures" / 'Hour_Season_Plot.pdf'
    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()

# main
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    df_de, df_uk = load_data()
    df_de, df_uk = preprocess(df_de, df_uk)

    # Load regional data 
    regions_de_path = BASE_DIR / "data" / "preprocessed" / "germany" /"geofiles"/ "Germany_merged.geojson"
    regions_uk_path = BASE_DIR / "data" / "preprocessed" / "uk" / "geofiles" / "UK_merged.geojson"

    regions_de_gdf = gpd.read_file(regions_de_path)
    regions_uk_gdf = gpd.read_file(regions_uk_path)


    # Compute normalized hourly rates
    norm_de = normalize_hourly(df_de, regions_de_gdf)
    norm_uk = normalize_hourly(df_uk, regions_uk_gdf)

    # Generate Plot 
    plot_hourly_clock(df_de, df_uk, regions_de_gdf, regions_uk_gdf)