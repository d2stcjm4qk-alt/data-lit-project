import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import plotly.express as px
import os
from pyrosm import OSM
import geopandas as gpd


def process_traffic_counts_df(
        df,
        region_geojson_path,
        lon_col,
        lat_col,
        traffic_col,
        length_col,
        road_class_col,
        region_code_col="region_code"
):
    df = df.copy()

    # numeric conversion
    df[traffic_col] = (
        df[traffic_col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df[length_col] = (
        df[length_col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df[traffic_col] = pd.to_numeric(df[traffic_col], errors="coerce")
    df[length_col] = pd.to_numeric(df[length_col], errors="coerce")
    df = df.dropna(subset=[traffic_col, length_col, lon_col, lat_col, road_class_col])

    # points
    geometry = gpd.points_from_xy(df[lon_col], df[lat_col])
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # regions
    gdf_regions = gpd.read_file(region_geojson_path).to_crs("EPSG:4326")

    # spatial join
    gdf_joined = gpd.sjoin(gdf_points, gdf_regions, how="left", predicate="within")
    gdf_joined = gdf_joined.dropna(subset=[region_code_col, traffic_col, length_col])

    # length-weighted AADF per region overall
    gdf_joined["weighted_traffic"] = gdf_joined[traffic_col] * gdf_joined[length_col]
    region_totals = (
        gdf_joined.groupby(region_code_col)
        .agg(
            weighted_sum=("weighted_traffic", "sum"),
            total_length_km=(length_col, "sum"),
            AADF_region_mean=(traffic_col, "mean"),
            n_points=(traffic_col, "count")
        ).reset_index()
    )
    region_totals["AADF_region_weighted"] = region_totals["weighted_sum"] / region_totals["total_length_km"]

    # length-weighted AADF per class
    class_table = (
        gdf_joined.groupby([region_code_col, road_class_col])
        .apply(lambda x: (x[traffic_col] * x[length_col]).sum() / x[length_col].sum())
        .reset_index()
        .rename(columns={0: "AADF_class"})
    )

    # pivot so each class is a column
    class_pivot = class_table.pivot(index=region_code_col, columns=road_class_col, values="AADF_class").reset_index()
    class_pivot.columns = [region_code_col if c == region_code_col else f"AADF_{c}" for c in class_pivot.columns]

    # merge back
    gdf_regions = gdf_regions.merge(region_totals[
                                        [region_code_col, "AADF_region_weighted", "AADF_region_mean", "total_length_km",
                                         "n_points"]
                                    ], on=region_code_col, how="left")

    gdf_regions = gdf_regions.merge(class_pivot, on=region_code_col, how="left")

    return gdf_regions


def load_ab_roads_from_pbf(pbf_path, country="DE"):
    osm = OSM(pbf_path)

    # 1. Load full driving network (no filters)
    roads = osm.get_network(
        network_type="driving",
        extra_attributes=["highway"]
    )

    # 2. Define classification mapping
    if country == "DE":
        valid_highways = [
            "motorway", "motorway_link",
            "trunk", "trunk_link",
            "primary", "primary_link"
        ]

        class_map = {
            "motorway": "A",
            "motorway_link": "A",
            "trunk": "B",
            "trunk_link": "B",
            "primary": "B",
            "primary_link": "B",
        }

    elif country == "UK":
        valid_highways = [
            "motorway", "motorway_link",
            "trunk", "trunk_link",
            "primary", "primary_link"
        ]

        class_map = {
            "motorway": "A",
            "motorway_link": "A",
            "trunk": "B",
            "trunk_link": "B",
            "primary": "B",
            "primary_link": "B",
        }

    # 3. Filter only A + B roads
    roads = roads[roads["highway"].isin(valid_highways)].copy()

    # 4. Project to meters and compute length
    roads = roads.to_crs(3857)
    roads["length_km"] = roads.length / 1000

    # 5. Assign road class
    roads["road_class"] = roads["highway"].map(class_map)

    return roads[["road_class", "length_km", "geometry"]]


def add_osm_road_lengths_by_region_single_download(
        gdf_regions,
        osm_roads,
        region_code_col="region_code"
):
    import geopandas as gpd
    import pandas as pd
    from tqdm import tqdm

    gdf_regions = gdf_regions.to_crs(3857).copy()
    osm_roads = osm_roads.to_crs(3857).copy()

    results = []

    for _, row in tqdm(
            gdf_regions.iterrows(),
            total=len(gdf_regions),
            desc="Clipping OSM roads to regions"
    ):
        region_code = row[region_code_col]
        geom = row.geometry

        clipped = osm_roads.clip(geom)

        length_A = clipped.loc[
            clipped["road_class"] == "A", "length_km"
        ].sum()

        length_B = clipped.loc[
            clipped["road_class"] == "B", "length_km"
        ].sum()

        results.append({
            region_code_col: region_code,
            "osm_length_A_km": float(length_A),
            "osm_length_B_km": float(length_B),
            "osm_total_length_km": float(length_A + length_B)
        })

    length_df = pd.DataFrame(results)

    gdf_regions = gdf_regions.merge(
        length_df,
        on=region_code_col,
        how="left"
    )

    return gdf_regions


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
    fig.show()  # Automatically opens browser


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent

    # --- Germany example ---
    ger_df = pd.read_csv(
        BASE_DIR / "data" / "raw" / "Germany" / "traffic" / "Jawe2023.csv",
        delimiter=";",
        encoding="cp1252",
        low_memory=False
    )

    # Example: filter to MobisSo only
    ger_df["Koor_WGS84_E"] = ger_df["Koor_WGS84_E"].astype(str).str.replace(",", ".").astype(float)
    ger_df["Koor_WGS84_N"] = ger_df["Koor_WGS84_N"].astype(str).str.replace(",", ".").astype(float)

    ger_df["DTV_Kfz_MobisSo_Q"] = (
        ger_df["DTV_Kfz_MobisSo_Q"]
        .astype(str)
        .str.replace(".", "", regex=False)  # remove thousands separator
        .str.replace(",", ".", regex=False)  # convert decimal separator
        .astype(float)
    )

    ger_gdf_traffic = process_traffic_counts_df(
        df=ger_df,
        region_geojson_path=BASE_DIR / "data" / "processed" / "geo_data" / "Germany_merged.geojson",
        lon_col="Koor_WGS84_E",
        lat_col="Koor_WGS84_N",
        traffic_col="DTV_Kfz_MobisSo_Q",
        length_col="Betriebs_km",
        road_class_col="Str_Kl",
        region_code_col="region_code"
    )

    ger_roads = load_ab_roads_from_pbf(
        r"D:\germany-latest.osm.pbf", country="DE"
    )

    uk_roads = load_ab_roads_from_pbf(
        r"D:\united-kingdom-251202.osm.pbf", country="UK"
    )

    ger_gdf = add_osm_road_lengths_by_region_single_download(
        ger_gdf_traffic, ger_roads, region_code_col="region_code"
    )

    plot_interactive_plotly(
        ger_gdf,
        value_col="AADF_B",
        name_col="region_code"
    )

    # --- UK example ---
    os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

    uk_df = pd.read_csv(
        BASE_DIR / "data" / "raw" / "uk" / "traffic" / "dft_traffic_counts_aadf.csv",
        delimiter=",",
        low_memory=False
    )

    # Example: filter to recent years or specific vehicle types
    # Filter to 2024 only
    uk_df_filtered = uk_df[uk_df["year"] == 2024]

    uk_to_german_map = {
        "PM": "A",  # Motorway
        "TM": "A",  # Motorway
        "PA": "B",  # Principal A road
        "TA": "B",  # Trunk A road
        "MB": None,  # minor, can ignore
        "MCU": None,  # minor / C / unclassified
        "M": None  # minor
    }

    # Apply to your dataframe
    uk_df_filtered["Str_Kl"] = uk_df_filtered["road_category"].map(uk_to_german_map)
    # Drop rows with Str_Kl == None
    uk_df_filtered = uk_df_filtered.dropna(subset=["Str_Kl"])

    uk_gdf_traffic = process_traffic_counts_df(
        df=uk_df_filtered,
        region_geojson_path=BASE_DIR / "data" / "processed" / "geo_data" / "UK_merged.geojson",
        lon_col="longitude",
        lat_col="latitude",
        traffic_col="all_motor_vehicles",
        length_col="link_length_km",
        road_class_col="Str_Kl",
        region_code_col="region_code"
    )

    max_value = uk_df_filtered['all_motor_vehicles'].max()
    print(f"Maximum all_motor_vehicles: {max_value}")
    print(f'length: {len(uk_df_filtered)}')

    uk_gdf = add_osm_road_lengths_by_region_single_download(
        uk_gdf_traffic, uk_roads, region_code_col="region_ons_code"
    )

    # Simplify geometries for plotting
    uk_gdf["geometry"] = uk_gdf["geometry"].simplify(0.0001, preserve_topology=True)
    # (uk_gdf_traffic)
    plot_interactive_plotly(uk_gdf, "AADF_B", "region_code")


if __name__ == "__main__":
    main()
