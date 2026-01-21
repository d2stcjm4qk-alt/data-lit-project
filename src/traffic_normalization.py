from pathlib import Path
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import calendar
from hierarchical_seasonal_analysis import AccidentDataProcessor


def process_country(gdf, name):
    df = gdf.copy()
    # Compute Weighted AADF
    total_len = df["osm_total_length_km"].replace(0, np.nan)
    df["AADF_combined"] = (
            (df["AADF_A"].fillna(0) * df["osm_length_A_km"].fillna(0) +
             df["AADF_B"].fillna(0) * df["osm_length_B_km"].fillna(0)) / total_len
    )

    # Exposure calculation
    df["exposure_annual"] = (df["AADF_combined"].fillna(0) * df["osm_total_length_km"]).clip(lower=0)
    df["days_in_month"] = df["month"].apply(lambda m: calendar.monthrange(2023, m)[1])
    df["exposure"] = df["exposure_annual"] * df["days_in_month"]

    return pd.DataFrame({
        "region_id": df["region_code"].astype(str),
        "month": df["month"].astype(int),
        "accident_count": df["accident_count"].astype(int),
        "exposure": df["exposure"],
        "country": name
    })


def normalize_accident_count(df, rate_per=1e6):
    """
    Normalize accident counts by exposure.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: 'accident_count' and 'exposure'
    rate_per : float, optional
        Scaling factor for the rate (default: per 1,000,000 exposure units)

    Returns
    -------
    pandas.DataFrame
        Original DataFrame with an added 'accident_rate' column
    """
    out = df.copy()

    out["accident_rate"] = (
                                   out["accident_count"]
                                   / out["exposure"].replace(0, np.nan)
                           ) * rate_per

    return out


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

    proc = AccidentDataProcessor()

    # Germany
    ger_regions = gpd.read_file(BASE_DIR / "data/processed/germany/traffic/ger_gdf_with_osm_roads.gpkg")
    ger_acc = proc.load_accidents(BASE_DIR / "data/processed/reduced_uk_dataset/modified_ger.csv",
                                  category_filters={"casualty_severity": [1]})
    ger_merged = proc.aggregate_by_region_monthly(ger_regions, ger_acc)
    ger_traffic_exposure_gdf = process_country(ger_merged, 'Germany')
    ger_normalized_by_traffic = normalize_accident_count(ger_traffic_exposure_gdf, 1e9)

    print(ger_normalized_by_traffic)


if __name__ == "__main__":
    main()
