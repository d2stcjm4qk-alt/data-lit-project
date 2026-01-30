import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
from pyrosm import OSM
from tqdm import tqdm
import plotly.express as px
from pathlib import Path


class TrafficInfrastructurePipeline:
    """
    Modular pipeline to harmonize AADF point data with OSM network lengths
    across synthetic 500k-population regions for Germany and the UK.
    """

    def __init__(self, region_path: Path, country_code: str):
        self.country_code = country_code.upper()
        self.regions = gpd.read_file(region_path).to_crs("EPSG:4326")
        self.metric_crs = "EPSG:3857"

    def process_point_traffic(self, df: pd.DataFrame, config: dict) -> gpd.GeoDataFrame:
        """Cleans raw geofiles CSVs and performs spatial aggregation."""
        df = df.copy()
        t_col, l_col = config['traffic_col'], config['length_col']
        lon_col, lat_col = config['lon'], config['lat']  # Get coordinate names

        # Numeric Sanitization for Traffic and Length
        for col in [t_col, l_col]:
            df[col] = (df[col].astype(str)
                       .str.replace(r"[.\s]", "", regex=True)
                       .str.replace(",", ".", regex=False))
            df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in [lon_col, lat_col]:
            df[col] = (df[col].astype(str)
                       .str.replace(",", ".", regex=False))  # Convert '10,18' to '10.18'
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=[t_col, l_col, lon_col, lat_col, config['class_col']])
        temp_regions = self.regions.copy()
        if "region_code" not in temp_regions.columns:
            temp_regions = temp_regions.reset_index()

        points = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326"
        )
        joined = gpd.sjoin(points, temp_regions, how="left", predicate="within")

        joined["weighted_traffic"] = joined[t_col] * joined[l_col]
        agg_results = joined.groupby("region_code").agg(
            weighted_sum_val=("weighted_traffic", "sum"),
            total_len_val=(l_col, "sum")
        ).reset_index()
        agg_results["AADF_region_weighted"] = agg_results["weighted_sum_val"] / agg_results["total_len_val"]

        class_pivot = joined.groupby(["region_code", config['class_col']]).apply(
            lambda x: (x[t_col] * x[l_col]).sum() / x[l_col].sum(), include_groups=False
        ).unstack().add_prefix("AADF_").reset_index()

        return self.regions.merge(agg_results[['region_code', 'AADF_region_weighted']], on="region_code",
                                  how="left").merge(class_pivot, on="region_code", how="left")

    def aggregate_osm_network(self, pbf_path: Path, gdf_target: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Loads OSM network from PBF and clips segments to regional boundaries."""
        osm = OSM(str(pbf_path))
        roads = osm.get_network(network_type="driving", extra_attributes=["highway"])

        class_map = {
            "motorway": "A", "motorway_link": "A",
            "trunk": "B", "trunk_link": "B", "primary": "B", "primary_link": "B"
        }

        roads = roads.to_crs(self.metric_crs)
        roads["road_class"] = roads["highway"].map(class_map)
        roads["length_km"] = roads.geometry.length / 1000

        regions_metric = gdf_target.to_crs(self.metric_crs)
        results = []
        for _, region in tqdm(regions_metric.iterrows(), total=len(regions_metric),
                              desc=f"Clipping OSM {self.country_code}"):
            clipped = roads.clip(region.geometry)
            results.append({
                "region_code": region.region_code,
                "osm_length_A_km": clipped[clipped.road_class == "A"].length_km.sum(),
                "osm_length_B_km": clipped[clipped.road_class == "B"].length_km.sum()
            })

        len_df = pd.DataFrame(results)
        len_df["osm_total_length_km"] = len_df["osm_length_A_km"] + len_df["osm_length_B_km"]
        return gdf_target.merge(len_df, on="region_code", how="left")


def validate_plot(gdf: gpd.GeoDataFrame, country: str):
    """Generates an interactive Plotly map for visual validation."""
    # Ensure WGS84 for Plotly
    gdf_plot = gdf.to_crs(4326).copy()

    # Identify which geofiles column to show (prefers B roads as they usually have more spatial variation)
    color_col = "AADF_B" if "AADF_B" in gdf_plot.columns else "AADF_region_weighted"

    fig = px.choropleth(
        gdf_plot,
        geojson=gdf_plot.geometry,
        locations=gdf_plot.index,
        color=color_col,
        hover_data=["region_code", "population", "osm_total_length_km", color_col],
        title=f"Validation Map: {country} Traffic Intensity ({color_col})"
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.show()


# --- EXECUTION ---

if __name__ == "__main__":

    # Remove the size limit for GeoJSON features globally
    os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"
    BASE = Path(__file__).resolve().parent.parent

    configs = {
        "germany": {
            "csv": BASE / "data/raw/Germany/geofiles/Jawe2023.csv",
            "pbf": BASE / "data/raw/Germany/geofiles/germany_ab_osmium.osm.pbf",
            "geo": BASE / "data/preprocessed/germany/geofiles/Germany_merged.geojson",
            "sep": ";", "enc": "cp1252",
            "traffic_col": "DTV_Kfz_MobisSo_Q", "length_col": "Betriebs_km",
            "lon": "Koor_WGS84_E", "lat": "Koor_WGS84_N", "class_col": "Str_Kl"
        },
        "UK": {
            "csv": BASE / "data/raw/uk/geofiles/dft_traffic_counts_aadf.csv",
            "pbf": BASE / "data/raw/uk/geofiles/uk_ab_osmium.osm.pbf",
            "geo": BASE / "data/preprocessed/uk/geofiles/UK_merged.geojson",
            "sep": ",", "enc": "utf-8",
            "traffic_col": "all_motor_vehicles", "length_col": "link_length_km",
            "lon": "longitude", "lat": "latitude", "class_col": "Str_Kl"
        }
    }

    for country, cfg in configs.items():
        print(f"\n--- Running Pipeline: {country} ---")
        df = pd.read_csv(cfg['csv'], sep=cfg['sep'], encoding=cfg['enc'], low_memory=False)

        if country == "UK":
            df = df[df["year"] == 2024].copy()
            df["Str_Kl"] = df["road_category"].map({"PM": "A", "TM": "A", "PA": "B", "TA": "B"})
            df = df.dropna(subset=["Str_Kl"])

        pipe = TrafficInfrastructurePipeline(cfg['geo'], country)
        gdf_traffic = pipe.process_point_traffic(df, cfg)
        final_gdf = pipe.aggregate_osm_network(cfg['pbf'], gdf_traffic)

        # Validation Plot
        validate_plot(final_gdf, country)

        # Save output
        if country.lower() == 'germany':
            country_abbrevation = 'ger'
        elif country.lower() == 'uk':
            country_abbrevation = 'uk'
        out = BASE / f"data/preprocessed/{country.lower()}/geofiles/{country_abbrevation}_gdf_with_osm_roads.gpkg"
        out.parent.mkdir(parents=True, exist_ok=True)
        final_gdf.to_file(out, driver="GPKG")
