# spatial_region_processing.py (refactored for manual per-region finer splits)

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from libpysal import weights
import os
import itertools
import numpy as np
import heapq
from typing import Optional, Dict, Tuple
import matplotlib.pyplot as plt
import contextily as ctx
import pyproj
import matplotlib.colors as mcolors
from pathlib import Path


##############################################
# LOADERS
##############################################

def load_boundaries(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs(3857)
    return gdf


def load_population_table(pop_path: str,
                          sheet_name=0,
                          skiprows=0,
                          code_col_keywords=("Regionalschlüssel", "Area Code", "Code"),
                          pop_col_keywords=("Bevölkerung", "population", "All ages")):
    if pop_path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(pop_path, sheet_name=sheet_name, skiprows=skiprows)
    else:
        df = pd.read_csv(pop_path, sep=";", low_memory=False)

    df.columns = [str(c).strip() for c in df.columns]

    code_col = next((c for c in df.columns if any(k.lower() in c.lower() for k in code_col_keywords)), None)
    pop_col = next((c for c in df.columns if any(k.lower() in c.lower() for k in pop_col_keywords)), None)

    if code_col is None or pop_col is None:
        raise ValueError("Could not detect code or population column")

    df = df[[code_col, pop_col]].copy()
    df.columns = ["region_code", "population"]

    df["region_code"] = df["region_code"].astype(str).str.strip()
    df["population"] = (
        df["population"].astype(str)
        .str.replace(" ", "")
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )
    return df


##############################################
# 2. ATTACH POPULATION
##############################################
def attach_population(
        regions_gdf: gpd.GeoDataFrame,
        pop_df: pd.DataFrame,
        region_code_col: str = "AGS"
) -> gpd.GeoDataFrame:
    # print(f'ääääääää: {(regions_gdf[region_code_col])}')
    # print(f'ööööööö: {(pop_df["region_code"])}')
    regions_gdf = regions_gdf.merge(
        pop_df, left_on=region_code_col, right_on="region_code", how="left"
    )
    # regions_gdf["population"] = regions_gdf["population"].fillna(0)
    # print(f'inside add piop: {regions_gdf["population"]}')
    missing = regions_gdf["population"].isna().sum()
    if missing:
        print(f"WARNING: {missing} regions have no population data.")

    return regions_gdf


##############################################
# 3. REPLACE REGION WITH MANUAL FINER UNITS
##############################################

def replace_region_with_finer_units(
        base_gdf: gpd.GeoDataFrame,
        finer_gdf: gpd.GeoDataFrame,
        region_id_col: str = "region_code",
        prefix_length: int = 2
) -> gpd.GeoDataFrame:
    """
    Replace a region in base_gdf with finer units based on the shared prefix
    of the region codes (default: first two digits).

    Example:
        Base: 11000 (Berlin)
        Finer: 11001–11012 (Bezirke)
        → Removes 11000 and inserts all finer units.
    """

    # Ensure all IDs are string
    base_gdf[region_id_col] = base_gdf[region_id_col].astype(str)
    finer_gdf[region_id_col] = finer_gdf[region_id_col].astype(str)

    # Determine prefix from finer_gdf (first two digits of its first row)
    prefix = finer_gdf[region_id_col].iloc[0][:prefix_length]

    # Mask: all base entries starting with that prefix
    mask = base_gdf[region_id_col].str[:prefix_length] == prefix

    if not mask.any():
        raise ValueError(f"No base regions found with prefix '{prefix}'")

    # Remove matched region(s)
    remaining = base_gdf[~mask]

    # Combine with finer units
    combined = pd.concat([remaining, finer_gdf], ignore_index=True)

    return gpd.GeoDataFrame(combined, crs=base_gdf.crs)


##############################################
# 4. MERGE SMALL REGIONS
##############################################

def merge_small_regions(
        gdf: gpd.GeoDataFrame,
        pop_col: str = "population",
        id_col: str = "region_code",
        target_population: int = 300_000,
        max_passes: int = 5,
        remove_below: int | None = None,
        remove_above: int | None = None,
        lookahead_depth: int = 2
) -> gpd.GeoDataFrame:
    """
    Merge regions toward a target population using lookahead neighbors
    and post-adjustment splitting for better population balance.

    Parameters:
        gdf: GeoDataFrame of regions
        pop_col: Name of the population column
        id_col: Region identifier column
        target_population: Desired population per merged region
        max_passes: Number of merging passes
        remove_below: Remove regions below this population
        remove_above: Remove regions above this population
        lookahead_depth: How many neighbor layers to consider in merging

    Returns:
        GeoDataFrame with merged regions
    """
    gdf = gdf.reset_index(drop=True).copy()
    if id_col not in gdf.columns:
        gdf[id_col] = gdf.index.astype(str)

    for _pass in range(max_passes):
        w = weights.Queen.from_dataframe(gdf)
        neighbors = {k: set(v) for k, v in w.neighbors.items()}

        visited = set()
        merged_output = []
        merged_indices = set()

        # Sort by smallest population first
        sorted_indices = gdf.sort_values(pop_col).index

        for idx in sorted_indices:
            if idx in visited:
                continue

            group = {idx}
            total_pop = float(gdf.loc[idx, pop_col])
            geom = gdf.loc[idx].geometry
            ids = [str(gdf.loc[idx, id_col])]

            # frontier with lookahead
            frontier = neighbors[idx].copy()
            for _ in range(lookahead_depth - 1):
                new_frontier = set()
                for f in frontier:
                    new_frontier.update(neighbors.get(f, []))
                frontier.update(new_frontier)
            frontier -= group

            improved = True
            while improved and frontier:
                candidates = [n for n in frontier if n not in visited and n not in group]

                if not candidates:
                    break

                # Pick candidate that brings total closest to target
                best_candidate = min(
                    candidates,
                    key=lambda n: abs((total_pop + gdf.loc[n, pop_col]) - target_population)
                )

                new_total = total_pop + gdf.loc[best_candidate, pop_col]

                # Stop if adding neighbor worsens distance to target
                if abs(new_total - target_population) >= abs(total_pop - target_population):
                    improved = False
                    break

                group.add(best_candidate)
                total_pop = new_total
                geom = unary_union([geom, gdf.loc[best_candidate].geometry])
                ids.append(str(gdf.loc[best_candidate, id_col]))

                # Expand frontier with lookahead again
                new_frontier = neighbors[best_candidate].copy()
                for _ in range(lookahead_depth - 1):
                    deeper = set()
                    for f in new_frontier:
                        deeper.update(neighbors.get(f, []))
                    new_frontier.update(deeper)
                frontier.update(new_frontier)
                frontier -= group

            visited.update(group)
            merged_indices.update(group)
            merged_output.append({
                id_col: "_".join(ids),
                pop_col: total_pop,
                "geometry": geom
            })

        # Add untouched regions
        untouched = set(gdf.index) - merged_indices
        for idx in untouched:
            merged_output.append({
                id_col: gdf.loc[idx, id_col],
                pop_col: float(gdf.loc[idx, pop_col]),
                "geometry": gdf.loc[idx, "geometry"]
            })

        gdf = gpd.GeoDataFrame(merged_output, crs=gdf.crs)

    # Post-adjustment: split large regions roughly
    if remove_above is not None:
        split_list = []
        for _, row in gdf.iterrows():
            if row[pop_col] > remove_above:
                num_splits = int(np.ceil(row[pop_col] / target_population))
                split_pop = row[pop_col] / num_splits
                for i in range(num_splits):
                    split_list.append({
                        id_col: f"{row[id_col]}_{i + 1}",
                        pop_col: split_pop,
                        "geometry": row["geometry"]
                    })
            else:
                split_list.append(row)
        gdf = gpd.GeoDataFrame(split_list, crs=gdf.crs)

    # Remove regions below remove_below
    if remove_below is not None:
        gdf = gdf[gdf[pop_col] >= remove_below].reset_index(drop=True)

    return gdf


##############################################
# 5. EXAMPLE PIPELINE
##############################################

def build_region_layer(boundary_path: str,
                       population_path: str,
                       finer_splits: dict | None = None,
                       min_population: int = 150000,
                       remove_below: int | None = None) -> gpd.GeoDataFrame:
    gdf = load_boundaries(boundary_path)
    pop_df = load_population_table(population_path)
    gdf = attach_population(gdf, pop_df)

    if finer_splits:
        for region_id, finer_path in finer_splits.items():
            finer_gdf = load_boundaries(finer_path)
            gdf = replace_region_with_finer_units(gdf, region_id, finer_gdf)

    gdf = merge_small_regions(gdf, min_population=min_population, remove_below=remove_below)

    return gdf


def load_district_population(
        file_path,
        sheet=0,
        header_row=0,
        district_col=0,
        population_col=2,
        start_code=19162,
        bez_width=2,
        nrows=None,
        csv_sep=","
):
    """
    Load district population data from Excel and produce a BEZ column
    that matches shapefile district codes. Can limit the number of rows.

    Parameters:
        file_path (str): Excel file path.
        sheet (int or str): Sheet index or name.
        header_row (int): Row index (0-based) containing column names.
        district_col (int or str): Column index or name for district.
        population_col (int or str): Column index or name for total population.
        start_code (int): Starting 5-digit regional code.
        bez_width (int): Number of digits for shapefile BEZ codes. Default 2.
        nrows (int, optional): Number of rows to read. Default None (all rows).

    Returns:
        pd.DataFrame: ['district', 'population', 'region_code']
    """
    # Load relevant columns
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(
            file_path,
            sheet_name=sheet,
            header=header_row,
            usecols=[district_col, population_col],
            nrows=nrows
        )
    elif ext == ".csv":
        df = pd.read_csv(
            file_path,
            sep=csv_sep,
            header=header_row,
            usecols=[district_col, population_col],
            nrows=nrows
        )
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    pd.set_option("display.max_rows", None)

    # Show all columns
    pd.set_option("display.max_columns", None)

    # Show full column width (no truncation)
    pd.set_option("display.max_colwidth", None)

    df.columns = ['district', 'population']

    # Clean population numbers
    df['population'] = pd.to_numeric(
        df['population'].astype(str).str.replace(" ", ""),
        errors='coerce'
    )
    # Drop rows with NaN population
    # df = df.dropna(subset=['population'])

    # Drop total row if present
    df = df[df['district'].str.lower() != 'berlin']

    df['population'] = df['population'].astype(int)

    # Assign sequential 5-digit regional codes (keep full code)
    # Assign sequential 5-digit regional codes starting from start_code
    df['region_code'] = [str(start_code + i).zfill(5) for i in range(len(df))]

    # Generate shapefile-compatible BEZ codes (last 'bez_width' digits)
    # df['region_code'] = df['region_code'].apply(lambda x: str(x)[-bez_width:]).str.zfill(bez_width)

    # Ensure BEZ is string and trimmed
    df['region_code'] = df['region_code'].astype(str).str.strip()
    print(f'berlin region: {df["region_code"]}')
    return df


def plot_population(
        regions_gdf: gpd.GeoDataFrame,
        title: str = "Population by Region",
        save_path: Optional[str] = None,
        prominent_cities: Optional[Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
        simplify_tolerance: Optional[float] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
):
    """
    Plot population per region with optional prominent city markers using LaTeX font and log scale.

    Args:
        regions_gdf: GeoDataFrame with geometries and population column.
        title: Plot title.
        save_path: If provided, saves the plot to this file.
        prominent_cities: Dict mapping city names to ((lon, lat), (dx, dy)) offsets.
        simplify_tolerance: If provided, simplifies geometries to reduce memory.
        vmin: Minimum value for color scale (log scale).
        vmax: Maximum value for color scale (log scale).
    """
    # Use LaTeX for all text
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16
    })

    gdf_plot = regions_gdf.copy()

    # Simplify geometries if requested
    if simplify_tolerance is not None:
        gdf_plot["geometry"] = gdf_plot["geometry"].simplify(tolerance=simplify_tolerance)

    # Project regions to Web Mercator for plotting
    gdf_plot = gdf_plot.to_crs(epsg=3857)

    column = "population"
    legend_label = "Population"

    # Determine fixed vmin and vmax for log scale
    if vmin is None:
        vmin = gdf_plot[column].min()
    if vmax is None:
        vmax = gdf_plot[column].max()

    norm = mcolors.LogNorm(vmin=max(vmin, 1), vmax=vmax)  # avoid log(0)

    fig, ax = plt.subplots(figsize=(12, 14))
    gdf_plot.plot(
        column=column,
        cmap="Blues",
        legend=True,
        ax=ax,
        edgecolor="gray",
        linewidth=0.3,
        alpha=0.8,
        norm=norm,
        legend_kwds={'label': legend_label, 'shrink': 0.6}
    )

    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=7, alpha=0.5)

    # Plot prominent cities if provided
    if prominent_cities:
        project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        for label, ((lon, lat), (dx, dy)) in prominent_cities.items():
            x, y = project.transform(lon, lat)
            ax.plot(x, y, "o", color='0.3', markersize=4)
            ax.text(
                x + dx, y + dy,
                rf"\textbf{{{label}}}",  # LaTeX bold
                fontsize=14,
                ha="center",
                va="bottom",
                color='0.3'
            )

    # ax.set_title(title)
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def uk_preprocessing(BASE_DIR):
    uk_shp_path = BASE_DIR / "data" / "raw" / "UK" / "regions" / "Local_Authority_Districts_May_2024_Boundaries_UK/LAD_MAY_2024_UK_BFE.shp"
    uk_pop_path = BASE_DIR / "data" / "raw" / "UK" / "population" / "mye24tablesuk.xlsx"

    # --- UK cities (lon/lat, offsets in meters) ---
    uk_cities = {
        "London": ((-0.1276, 51.5074), (0, 50000)),
        "Birmingham": ((-1.8998, 52.4862), (0, 25000)),
        "Glasgow": ((-4.2518, 55.8642), (0, 50000)),
        "Liverpool": ((-2.9779, 53.4084), (0, 20000)),
        "Bristol": ((-2.5879, 51.4545), (0, 25000)),
        "Edinburgh": ((-3.1883, 55.9533), (0, 25000))
    }

    uk_lad = load_boundaries(uk_shp_path)
    uk_pop = load_population_table(uk_pop_path, sheet_name="MYE2 - Persons", skiprows=7,
                                   code_col_keywords=("Code",), pop_col_keywords=("All ages",))
    uk_lad = attach_population(uk_lad, uk_pop, region_code_col="LAD24CD")
    # Merge small regions to target population
    merged_uk = merge_small_regions(uk_lad, pop_col="population", target_population=500_000,
                                    remove_below=150_000)
    merged_uk.to_file(BASE_DIR / "data" / "processed" / "geo_data" / "UK_merged.geojson",
                      driver="GeoJSON")

    plot_population(uk_lad, prominent_cities=uk_cities,
                    title="UK population of regions",
                    save_path=BASE_DIR / "results" / "figures" / "uk" / "UK_orig_regions_pop.png",
                    vmin=1e4,
                    vmax=4e6)

    plot_population(merged_uk, prominent_cities=uk_cities,
                    title="UK population of merged regions",
                    save_path=BASE_DIR / "results" / "figures" / "uk" / "UK_merged_regions_pop.png",
                    vmin=1e4,
                    vmax=4e6)


def main():
    # Base directory (project root, one level up from src)
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Paths to data
    germany_shp_path = BASE_DIR / "data" / "raw" / "Germany" / "regions" / "vg250-ew_12-31.utm32s.shape.ebenen/vg250-ew_12-31.utm32s.shape.ebenen/vg250-ew_ebenen_1231/VG250_KRS.shp"
    germany_pop_path = BASE_DIR / "data" / "raw" / "Germany" / "population" / "04-kreise.xlsx"

    df_berlin = load_district_population(
        file_path= BASE_DIR / "data" / "raw" / "Germany" / "population" / "SB_A01-05-00_2025h01_BE.xlsx",
        sheet=5,  # 5th sheet
        header_row=6,  # headers on row 8
        district_col=0,
        population_col=1,
        start_code=11001,
        bez_width=2,  # matches shapefile BEZ codes
        nrows=12
    )

    df_munich = load_district_population(
        file_path=BASE_DIR / "data" / "raw" / "Germany" / "population" / "bevolkerung_bezirke_neu.csv",
        sheet=0,  # 5th sheet
        header_row=0,  # headers on row 8
        district_col=1,
        population_col=2,
        start_code=19162,
        bez_width=2,  # matches shapefile BEZ codes
        nrows=25
    )

    # Optional: finer regions for specific cities
    finer_splits = [
        {
            "name": "Berlin",
            "region_code": "11000",
            "shape_path": BASE_DIR / "data" / "raw" / "Germany" / "regions" / "RBS_OD_BEZ_2015_12/RBS_OD_BEZ_2015_12.shp",
            # https://daten.berlin.de/datensaetze/rbs-bezirke-dezember-2015
            "population_df": df_berlin  # preloaded DataFrame with 'region_code' and 'population' columns
        },
        # Add other finer regions similarly:
        {
            "name": "Hamburg",
            "region_code": "02000",
            "shape_path": BASE_DIR / "data" / "raw" / "Germany" / "regions" / "HH_ALKIS_Stadtteile_2016_6864215073834709224/Hamburg_Stadtteilestatistik.shp",
            "population_df": None
        },
        {
            "name": "Munich",
            "region_code": "09162",
            "shape_path": BASE_DIR / "data" / "raw" / "Germany" / "regions" / "bezirke_muenchen_-8237817724309258366/bezirke_muenchen.shp",
            "population_df": df_munich
        }
    ]

    # Load base Germany boundaries
    kreise = load_boundaries(germany_shp_path)
    # Load population table
    pop_df = load_population_table(germany_pop_path, sheet_name=1, skiprows=1)
    # Quick fix if population has an extra 0
    pop_df['population'] = pop_df['population'] / 10
    pop_df['region_code'] = pop_df['region_code'].astype(str).str.strip()

    kreise = attach_population(kreise, pop_df)

    # --- Germany cities (lon/lat, offsets in meters) ---
    germany_cities = {
        "Berlin": ((13.4050, 52.5200), (0, 25000)),
        "Hamburg": ((9.9937, 53.5511), (0, 25000)),
        "Munich": ((11.5820, 48.1351), (0, 25000)),
        "Cologne": ((6.9603, 50.9375), (0, 25000)),
        "Frankfurt": ((8.6821, 50.1109), (0, 25000)),
        "Stuttgart": ((9.1829, 48.7758), (0, 25000)),
        "Leipzig": ((12.3731, 51.3397), (0, 25000))
    }

    plot_population(kreise, prominent_cities=germany_cities,
                    title="Germany population of regions",
                    save_path=BASE_DIR / "results" / "figures" / "germany" / "Germany_orig_regions_pop.png",
                    vmin=1e4,
                    vmax=4e6)

    list_of_finer_gdfs = []
    # Berlin setup
    berlin_gdf = load_boundaries(finer_splits[0]["shape_path"])
    berlin_gdf = berlin_gdf.dissolve(by="BEZ")
    berlin_gdf = berlin_gdf.reset_index()
    start_berlin_id = 11001
    berlin_gdf['BEZ'] = [
        f"{start_berlin_id + i:05d}" for i in range(len(berlin_gdf))
    ]

    # Attach manually preloaded population
    berlin_gdf = attach_population(berlin_gdf, finer_splits[0]["population_df"], region_code_col='BEZ')
    list_of_finer_gdfs.append(berlin_gdf)
    # Hamburg setup
    hamburg_gdf = load_boundaries(finer_splits[1]["shape_path"])
    hamburg_gdf["population"] = hamburg_gdf['Anzahl_der']
    start = 2000
    hamburg_gdf['region_code'] = [
        f"{start + i:05d}" for i in range(len(hamburg_gdf))
    ]

    hamburg_gdf['region_code'] = hamburg_gdf['region_code'].astype(str).str.strip()
    list_of_finer_gdfs.append(hamburg_gdf)

    # Region Munich
    munich_gdf = load_boundaries(finer_splits[2]["shape_path"])
    munich_gdf = munich_gdf.drop_duplicates(subset=["sb_nummer", "name"], keep="first")
    start_munich_id = 19162
    munich_gdf['BEZ'] = [
        f"{start_munich_id + i:05d}" for i in range(len(munich_gdf))
    ]
    kreise.loc[kreise['region_code'] == "09162", 'region_code'] = "19162"

    # Attach manually preloaded population
    munich_gdf = attach_population(munich_gdf, finer_splits[2]["population_df"], region_code_col='BEZ')
    list_of_finer_gdfs.append(munich_gdf)

    for finer in list_of_finer_gdfs:
        for region_id in finer["region_code"].unique():
            # print(region_id)
            kreise = replace_region_with_finer_units(
                base_gdf=kreise,
                finer_gdf=finer,
                region_id_col="region_code"
            )

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)

    # Remove Regions with too large population but no smaller regions available
    codes_to_remove = ["03241", "05315"]
    kreise = kreise[~kreise['region_code'].isin(codes_to_remove)]

    # Merge small regions to target population
    merged_ger = merge_small_regions(kreise, pop_col="population", target_population=500_000,
                                     remove_below=150_000)
    print(merged_ger['population'])

    import plotly.express as px

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

    # plot_interactive_plotly(merged_ger, "population", "region_code")

    plot_population(merged_ger, prominent_cities=germany_cities,
                    title="Germany population of merged regions",
                    save_path=BASE_DIR / "results" / "figures" / "germany" / "Germany_merged_regions_pop.png",
                    vmin=1e4,
                    vmax=4e6)

    uk_preprocessing(BASE_DIR)
    # Optionally save merged layer
    merged_ger.to_file(BASE_DIR / "data" / "processed" / "geo_data" / "Germany_merged.geojson",
                       driver="GeoJSON")

    print("Processing complete. Merged regions saved to data/kreise_merged.geojson")


if __name__ == "__main__":
    main()
