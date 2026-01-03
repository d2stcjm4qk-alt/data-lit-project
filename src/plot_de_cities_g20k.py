# plot_de_cities_gt20k.py
import os
import zipfile
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# ---------- KONFIG ----------
# Pfad zur heruntergeladenen GeoNames ZIP-Datei (cities5000.zip)
# Lade die ZIP hierher: https://download.geonames.org/export/dump/
geonames_zip = "cities5000.zip"  # passe an deinen Pfad an
geonames_txt = "cities5000.txt"  # Datei im ZIP

# Natural Earth: Länder-Admin 0 (10m). Wir lesen direkt vom NACIS CDN (funktioniert mit geopandas).
naturalearth_countries_url = (
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
)

POP_THRESHOLD = 20000
COUNTRY_ISO2 = "DE"
CRS_WGS84 = "EPSG:4326"
# --------------------------

# 1) Lade Natural Earth Länder
world = gpd.read_file(naturalearth_countries_url)
# Wähle Deutschland heraus (Name-Feld kann 'Germany' heißen)
de = world[world['name'].isin(["Germany", "Deutschland"])].copy()
if de.empty:
    # fallback: filter by ISO_A2 or ISO_A3 if name search nicht geklappt hat
    if 'iso_a2' in world.columns:
        de = world[world['iso_a2'] == 'DE'].copy()
    elif 'adm0_a3' in world.columns:
        de = world[world['adm0_a3'] == 'DEU'].copy()
de = de.to_crs(CRS_WGS84)

# 2) Extrahiere/lese GeoNames cities5000.txt aus ZIP
if not os.path.exists(geonames_zip):
    raise FileNotFoundError(
        f"ZIP-Datei '{geonames_zip}' nicht gefunden. Lade sie von https://download.geonames.org/export/dump/ herunter."
    )

with zipfile.ZipFile(geonames_zip, 'r') as z:
    if geonames_txt not in z.namelist():
        # manchmal heißt die Datei anders; suche nach passenden Dateien
        candidates = [n for n in z.namelist() if n.lower().startswith("cities")]
        if not candidates:
            raise FileNotFoundError("Keine passende 'cities...' Datei im ZIP gefunden.")
        geonames_txt = candidates[0]
    with z.open(geonames_txt) as f:
        # GeoNames columns according to readme:
        cols = [
            "geonameid", "name", "asciiname", "alternatenames",
            "latitude", "longitude", "feature_class", "feature_code",
            "country_code", "cc2", "admin1", "admin2", "admin3", "admin4",
            "population", "elevation", "dem", "timezone", "modification_date"
        ]
        df = pd.read_csv(
            f,
            sep='\t',
            header=None,
            names=cols,
            dtype={
                "country_code": str,
                "population": float
            },
            low_memory=False,
            na_values=['', 'NULL']
        )

# 3) Filter: Deutschland + population > 20000
df_de = df[(df['country_code'] == COUNTRY_ISO2) & (df['population'] > POP_THRESHOLD)].copy()

# Optional: sicherstellen, dass lat/lon vorhanden
df_de = df_de.dropna(subset=['latitude', 'longitude'])

# 4) In GeoDataFrame umwandeln
geometry = [Point(xy) for xy in zip(df_de['longitude'], df_de['latitude'])]
gdf_cities = gpd.GeoDataFrame(df_de, geometry=geometry, crs=CRS_WGS84)

# 5) Clip/Spatial join mit Deutschland-Polygon (sofern nötig)
# Manche GeoNames-Einträge an Grenzen könnten knapp außerhalb liegen; wir beschneiden
gdf_cities = gpd.sjoin(gdf_cities, de[['geometry']], how='inner', predicate='within')
# sjoin erzeugt index_right -> löschen
gdf_cities = gdf_cities.drop(columns=[c for c in gdf_cities.columns if c.startswith('index_')], errors='ignore')

# 6) Plotten
fig, ax = plt.subplots(1, 1, figsize=(10, 12))
de.plot(ax=ax, edgecolor='black', linewidth=0.6, facecolor='#f2f2f2')
# Punkte: Größe nach Population (sqrt skaliert für bessere Sichtbarkeit)
sizes = (gdf_cities['population'].astype(float).fillna(0).pow(0.5)) / 5.0
gdf_cities.plot(ax=ax, markersize=sizes, alpha=0.8)

# Optional: Beschriftungen für größere Städte (z.B. >100k)
for idx, row in gdf_cities[gdf_cities['population'] > 100000].iterrows():
    ax.annotate(text=row['name'], xy=(row.geometry.x, row.geometry.y),
                xytext=(3, 3), textcoords="offset points", fontsize=8)

ax.set_title(f"Deutsche Städte mit > {POP_THRESHOLD:,} Einwohnern (Quelle: GeoNames)")
ax.set_axis_off()

# 7) Speichern
plt.tight_layout()
out_png = "germany_cities_gt20k.png"
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"Karte gespeichert als: {out_png}")
plt.show()
