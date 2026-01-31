import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from tueplots import bundles

plt.rcParams.update(bundles.icml2024(column="full"))
plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["legend.title_fontsize"] = 10

BASE_DIR = Path(__file__).resolve().parent.parent

seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
road_types = ['Motorway', 'Other Roads']


def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    return None


os.environ['OGR_GEOJSON_MAX_OBJ_SIZE'] = '0'

uk_regions = gpd.read_file(BASE_DIR / "data" / "processed" / "geo_data" / "UK_merged.geojson")
de_regions = gpd.read_file(BASE_DIR / "data" / "preprocessed" / "geofiles" / "Germany_merged.geojson")

if uk_regions.crs is None:
    uk_regions = uk_regions.set_crs("EPSG:4326")
if de_regions.crs is None:
    de_regions = de_regions.set_crs("EPSG:4326")

uk_total_pop = uk_regions['population'].sum()
de_total_pop = de_regions['population'].sum()

data_dir = BASE_DIR / "data" / "preprocessed"
uk_df = pd.read_csv(data_dir / "uk" / "collisions" / "preprocessed_uk.csv" , low_memory=False)
de_df = pd.read_csv(data_dir / "germany" / "collisions" / "preprocessed_ger.csv" , low_memory=False)

uk_df['season'] = uk_df['month'].apply(get_season)
de_df['season'] = de_df['month'].apply(get_season)

uk_fatal = uk_df[uk_df['collision_severity'] == 1].copy()
de_fatal = de_df[de_df['casualty_severity'] == 1].copy()

if 'first_road_class' in uk_fatal.columns:
    uk_fatal['road_type'] = uk_fatal['first_road_class'].apply(
        lambda x: 'Motorway' if x == 1 else 'Other Roads'
    )

for col in ['latitude', 'longitude']:
    de_fatal[col] = de_fatal[col].astype(str).str.replace(',', '.').astype(float)

de_fatal_gdf = gpd.GeoDataFrame(
    de_fatal,
    geometry=gpd.points_from_xy(de_fatal.longitude, de_fatal.latitude),
    crs="EPSG:4326"
)

motorways = gpd.read_file(BASE_DIR / "data" / "processed" / "geo_data" / "germany_motorways.geojson")
de_fatal_gdf = de_fatal_gdf.to_crs(epsg=25832)
motorways = motorways.to_crs(epsg=25832)

motorway_union = motorways.geometry.values.union_all()


def classify_road(point, motorway_union, threshold_m=50):
    return "Motorway" if point.distance(motorway_union) <= threshold_m else "Other Roads"


de_fatal_gdf['road_type'] = de_fatal_gdf.geometry.apply(
    lambda pt: classify_road(pt, motorway_union)
)

de_fatal['road_type'] = de_fatal_gdf['road_type'].values

uk_results = []
for season in seasons:
    for road in road_types:
        mask = (uk_fatal['season'] == season) & (uk_fatal['road_type'] == road)
        count = mask.sum()
        rate = count / uk_total_pop * 100_000
        uk_results.append({
            'Season': season,
            'Road_Type': road,
            'Fatal_Count': count,
            'Rate_per_100k_Pop': rate
        })

uk_results_df = pd.DataFrame(uk_results)

de_results = []
for season in seasons:
    for road in road_types:
        mask = (de_fatal['season'] == season) & (de_fatal['road_type'] == road)
        count = mask.sum()
        rate = count / de_total_pop * 100_000
        de_results.append({
            'Season': season,
            'Road_Type': road,
            'Fatal_Count': count,
            'Rate_per_100k_Pop': rate
        })

de_results_df = pd.DataFrame(de_results)

uk_mot_count = uk_results_df[uk_results_df['Road_Type'] == 'Motorway']['Fatal_Count'].sum()
uk_oth_count = uk_results_df[uk_results_df['Road_Type'] == 'Other Roads']['Fatal_Count'].sum()
de_mot_count = de_results_df[de_results_df['Road_Type'] == 'Motorway']['Fatal_Count'].sum()
de_oth_count = de_results_df[de_results_df['Road_Type'] == 'Other Roads']['Fatal_Count'].sum()

combined_mot = uk_mot_count + de_mot_count
combined_oth = uk_oth_count + de_oth_count
total_combined = combined_mot + combined_oth

prop_motorway = (combined_mot / total_combined) * 100
prop_other = (combined_oth / total_combined) * 100

comparison_data = []
for season in seasons:
    uk_mot_s = uk_results_df[(uk_results_df['Season'] == season) & (uk_results_df['Road_Type'] == 'Motorway')][
        'Rate_per_100k_Pop'].sum()
    de_mot_s = de_results_df[(de_results_df['Season'] == season) & (de_results_df['Road_Type'] == 'Motorway')][
        'Rate_per_100k_Pop'].sum()
    uk_oth_s = uk_results_df[(uk_results_df['Season'] == season) & (uk_results_df['Road_Type'] == 'Other Roads')][
        'Rate_per_100k_Pop'].sum()
    de_oth_s = de_results_df[(de_results_df['Season'] == season) & (de_results_df['Road_Type'] == 'Other Roads')][
        'Rate_per_100k_Pop'].sum()

    comparison_data.append({
        'Season': season,
        'UK_Motorway': uk_mot_s,
        'DE_Motorway': de_mot_s,
        'Motorway_Ratio': de_mot_s / uk_mot_s if uk_mot_s > 0 else 0,
        'UK_Other': uk_oth_s,
        'DE_Other': de_oth_s,
        'Other_Ratio': de_oth_s / uk_oth_s if uk_oth_s > 0 else 0
    })

comparison_df = pd.DataFrame(comparison_data)

fig, ax1 = plt.subplots(constrained_layout=True)

x_pos = np.arange(2)
width = 0.5

bars_prop = ax1.bar(x_pos, [prop_motorway, prop_other], width,
                    color='#C5BFE0', alpha=0.18, zorder=1,
                    edgecolor='#9B8FCC', linewidth=1.2)

ax1.set_ylabel(r'Proportion (\%)', color='#999999')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([])
ax1.tick_params(axis='y', labelcolor='#999999', colors='#CCCCCC')
ax1.tick_params(axis='x', length=0)
ax1.grid(axis='y', alpha=0.08, zorder=0, linestyle='-', linewidth=0.8)
ax1.set_ylim(0, 105)
ax1.set_xlim(-0.6, 1.6)

for spine in ax1.spines.values():
    spine.set_alpha(0.2)
ax1.spines['top'].set_visible(False)

ax1.text(0, prop_motorway + 5, r'{:.1f}\%'.format(prop_motorway),
         ha='center', color='#999999', zorder=50)
ax1.text(1, prop_other + 5, r'{:.1f}\%'.format(prop_other),
         ha='center', color='#999999', zorder=50)

ax1.text(0, 3, r'Motorway', ha='center', va='center', color='#888888', zorder=150)
ax1.text(1, 3, r'Other Roads', ha='center', va='center', color='#888888', zorder=150)

ax2 = ax1.twinx()

mot_ratios = [comparison_df[comparison_df['Season'] == s]['Motorway_Ratio'].values[0] for s in seasons]
oth_ratios = [comparison_df[comparison_df['Season'] == s]['Other_Ratio'].values[0] for s in seasons]

season_x = np.linspace(-0.35, 1.35, 4)

ax2.plot(season_x, mot_ratios, linewidth=8, color='#E8577A', alpha=0.15, zorder=2)
line_mot = ax2.plot(season_x, mot_ratios, marker='o', linewidth=3.5, markersize=11,
                    color='#E8577A', zorder=4, alpha=0.95)[0]

ax2.plot(season_x, oth_ratios, linewidth=8, color='#4FB05C', alpha=0.15, zorder=2)
line_oth = ax2.plot(season_x, oth_ratios, marker='s', linewidth=3.5, markersize=10,
                    color='#4FB05C', zorder=4, alpha=0.95)[0]

for i, (x, mot_val, oth_val) in enumerate(zip(season_x, mot_ratios, oth_ratios)):
    x_offset = 0.08 if seasons[i] == 'Autumn' else 0
    ax2.text(x + x_offset, mot_val + 0.25, r'${:.2f}\times$'.format(mot_val),
             ha='center', color='#E8577A', fontweight='bold', fontsize=13)
    ax2.text(x, oth_val + 0.25, r'${:.2f}\times$'.format(oth_val),
             ha='center', color='#4FB05C', fontweight='bold', fontsize=13)

ax2.axhline(y=1.0, color='#666666', linestyle='--', linewidth=1.8, alpha=0.5, zorder=2)

ax2.set_ylabel(r'DE/UK Rate Ratio', fontweight='bold')
ax2.set_ylim(0.5, max(max(mot_ratios), max(oth_ratios)) * 1.35)
ax2.spines['right'].set_linewidth(2)
ax2.spines['right'].set_color('#444444')
ax2.spines['left'].set_alpha(0.2)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_alpha(0.2)

ax2.set_xlim(-0.6, 1.6)
ax2.set_xticks([])

for i, (x, season) in enumerate(zip(season_x, seasons)):
    ax2.text(x, -0.06, season, ha='center', fontsize=13,
             transform=ax2.get_xaxis_transform(),
             color='#222222', fontweight='bold')

ratio_handles = [line_mot, line_oth]
ratio_labels = [r'DE/UK ratio: Motorways', r'DE/UK ratio: Other Roads']

context_handles = [
    Patch(facecolor='#C5BFE0', alpha=0.18, edgecolor='#9B8FCC', linewidth=1.2),
    Line2D([0], [0], color='#666666', linestyle='--', linewidth=1.8, alpha=0.5)
]
context_labels = [r'Overall proportion (\%)', r'Equal rates (ratio = 1.0)']

legend = ax2.legend(ratio_handles + context_handles, ratio_labels + context_labels,
                    loc='upper left', framealpha=0.97, edgecolor='#AAAAAA',
                    title=r'Legend')
legend.get_frame().set_linewidth(1.5)

output_dir = BASE_DIR / "results"
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / 'season-motorway.pdf', dpi=300, bbox_inches='tight')
plt.show()
