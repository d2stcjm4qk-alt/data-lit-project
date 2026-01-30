import geopandas as gpd
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


class AccidentNormalizer:
    """
    Handles spatial joining of accident points to regions and
    calculates population-based normalization metrics.
    """

    def __init__(self, regions_gdf: gpd.GeoDataFrame):
        self.regions = regions_gdf
        if self.regions.crs is None:
            raise ValueError("Regions GeoDataFrame must have a defined CRS.")

    def load_accident_data(self, path: Path, filters: dict = None) -> gpd.GeoDataFrame:
        """Loads raw CSV, cleans coordinates, and converts to GeoDataFrame."""
        df = pd.read_csv(path, sep=",", dtype=str)

        # Clean and convert coordinates
        for col in ["longitude", "latitude"]:
            df[col] = df[col].str.replace(",", ".", regex=False).astype(float)

        # Apply filtering (e.g., fatality severity only)
        if filters:
            for col, allowed_values in filters.items():
                df = df[df[col].isin(allowed_values)]

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326"
        )
        return gdf.to_crs(self.regions.crs)

    def attach_regions_and_count(self, accidents_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Performs spatial join and aggregates accident counts per region."""
        # Spatial join
        joined = gpd.sjoin(accidents_gdf, self.regions, how="left", predicate="within")

        # Aggregate counts
        counts = joined.groupby("region_code").size().rename("accident_count").reset_index()

        # Merge back to master regions GeoDataFrame
        self.regions = self.regions.merge(counts, on="region_code", how="left")
        self.regions["accident_count"] = self.regions["accident_count"].fillna(0)

        return self.regions

    def normalize_by_population(self, scale: int = 100_000):
        """Calculates the normalized rate (default: per 100k inhabitants)."""
        self.regions[f"accidents_per_{scale // 1000}k"] = (
                self.regions["accident_count"] / self.regions["population"] * scale
        )
        return self.regions

    def plot_results(self, column: str = "accidents_per_100k", title: str = "Accident Distribution"):
        """Quick diagnostic plot to verify spatial join and normalization."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
        self.regions.plot(
            column=column,
            cmap="OrRd",
            legend=True,
            ax=ax,
            legend_kwds={'label': f"Rate ({column})", 'orientation': "horizontal"}
        )
        ax.set_title(title)
        ax.axis("off")
        plt.show()


# --- Example Usage for Reproducibility ---
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    # 1. Load your 500k-merged regions
    region_path = BASE_DIR / "data" / "preprocessed" / "germany" / "geofiles" / "Germany_merged.geojson"
    regions_gdf = gpd.read_file(region_path)

    # 2. Initialize Normalizer
    normalizer = AccidentNormalizer(regions_gdf)

    # 3. Process Accidents
    acc_path = BASE_DIR / "data" / "preprocessed" / "germany" / "collisions" / "preprocessed_ger.csv"
    acc_gdf = normalizer.load_accident_data(acc_path, filters={"casualty_severity": ["1"]})

    # 4. Generate Final Dataset
    final_regions = normalizer.attach_regions_and_count(acc_gdf)
    final_regions = normalizer.normalize_by_population(scale=100_000)

    # 5. Diagnostic Plot: Check if the map looks correct (no empty regions where expected)
    normalizer.plot_results(title="Fatal Accidents per 100k Inhabitants (Germany)")
