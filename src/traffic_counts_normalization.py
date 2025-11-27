import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

# Base directory (project root, one level up from src)
BASE_DIR = Path(__file__).resolve().parent.parent
zaehlstellen_csv =  BASE_DIR / "data" / "raw" / "Germany" / "traffic" / "Jawe2023.csv"
df_zaehlstellen = pd.read_csv(zaehlstellen_csv, delimiter=';', encoding='cp1252')

# Punkte-Geometrie erstellen
df_zaehlstellen['Koor_WGS84_E'] = df_zaehlstellen['Koor_WGS84_E'].str.replace(',', '.').astype(float)
df_zaehlstellen['Koor_WGS84_N'] = df_zaehlstellen['Koor_WGS84_N'].str.replace(',', '.').astype(float)

geometry = [Point(xy) for xy in zip(df_zaehlstellen['Koor_WGS84_E'], df_zaehlstellen['Koor_WGS84_N'])]
gdf_zaehlstellen = gpd.GeoDataFrame(df_zaehlstellen, geometry=geometry, crs="EPSG:4326")

# --- 2. Regionen mit Population laden ---
geojson_regions = BASE_DIR / "data" / "processed" / "geo_data" / "Germany_merged.geojson"
gdf_regions = gpd.read_file(geojson_regions)
gdf_regions = gdf_regions.to_crs("EPSG:4326")  # CRS angleichen

# --- 3. Zählstellen den Regionen zuordnen ---
gdf_joined = gpd.sjoin(gdf_zaehlstellen, gdf_regions, how="left", predicate='within')
print(gdf_regions.columns)
# --- 4. Aggregation der Verkehrsdaten pro Region ---
traffic_region = gdf_joined.groupby('region_code')['DTV_Kfz_MobisSo_Q'].mean().reset_index()

# --- 5. Verkehrsdaten zu den Regionen hinzufügen ---
gdf_regions = gdf_regions.merge(traffic_region, on='region_code', how='left')
print(gdf_regions.columns)
# --- 6. GeoJSON exportieren ---
#gdf_regions.to_file("regionen_mit_verkehr.geojson", driver='GeoJSON')
print("Fertig! GeoJSON mit Regionen, Population und Verkehr erstellt.")

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

plot_interactive_plotly(gdf_regions, "DTV_Kfz_MobisSo_Q", "region_code")
plot_interactive_plotly(gdf_regions, "population", "region_code")