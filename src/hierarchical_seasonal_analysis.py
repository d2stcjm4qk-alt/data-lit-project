import os
import calendar
from pathlib import Path
from typing import Optional, Dict, List, Union
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from tueplots import bundles




class AccidentDataProcessor:
    @staticmethod
    def load_accidents(
            csv_path: Union[str, Path],
            lon_col: str = "longitude",
            lat_col: str = "latitude",
            separator: str = ",",
            category_filters: Optional[Dict[str, List]] = None,
            target_crs: str = "EPSG:4326"
    ) -> gpd.GeoDataFrame:
        df = pd.read_csv(str(csv_path), sep=separator, low_memory=False)

        # Clean coordinates
        for col in [lon_col, lat_col]:
            df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

        if category_filters:
            for col, values in category_filters.items():
                if col in df.columns:
                    df = df[df[col].isin(values)]

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs="EPSG:4326"
        )
        return gdf.to_crs(target_crs) if target_crs != "EPSG:4326" else gdf

    @staticmethod
    def aggregate_by_region_monthly(
            regions_gdf: gpd.GeoDataFrame,
            accidents_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        regions_gdf = regions_gdf.to_crs(accidents_gdf.crs)

        joined = gpd.sjoin(accidents_gdf, regions_gdf, how="left", predicate="within")

        # Count and create complete grid (every region for every month)
        counts = joined.groupby(["index_right", "month"]).size().rename("accident_count").reset_index()

        full_grid = pd.MultiIndex.from_product(
            [regions_gdf.index, range(1, 13)],
            names=["index_right", "month"]
        ).to_frame(index=False)

        final_df = full_grid.merge(counts, on=["index_right", "month"], how="left")
        final_df["accident_count"] = final_df["accident_count"].fillna(0).astype(int)

        merged = final_df.merge(regions_gdf, left_on="index_right", right_index=True, how="left")
        return gpd.GeoDataFrame(merged, geometry="geometry", crs=regions_gdf.crs)


def build_bayes_dataset(germany_gdf, uk_gdf):
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

    return pd.concat([process_country(germany_gdf, "Germany"), process_country(uk_gdf, "UK")], ignore_index=True)

def build_bayes_dataset_population(germany_gdf, uk_gdf):
    def process_country(gdf, name):
        df = gdf.copy()

        return pd.DataFrame({
            "region_id": df["region_code"].astype(str),
            "month": df["month"].astype(int),
            "accident_count": df["accident_count"].astype(int),
            "exposure": df["population"],
            "country": name
        })

    return pd.concat([process_country(germany_gdf, "Germany"), process_country(uk_gdf, "UK")], ignore_index=True)


def hierarchical_poisson_model(accidents, exposure, country_idx, region_idx, month_idx, n_countries, n_regions,
                               n_months=12):
    # Country-level priors
    mu_country = numpyro.sample("mu_country", dist.Normal(1.0, 2.0).expand([n_countries]))
    sigma_country = numpyro.sample("sigma_country", dist.HalfNormal(1.5).expand([n_countries]))

    # Regional random effects (Centered on country base rate)
    region_raw = numpyro.sample("region_raw", dist.Normal(0.0, 1.0).expand([n_regions]))
    region_effect = sigma_country[country_idx] * region_raw[region_idx]

    # Seasonal effects - Country Level
    month_country = numpyro.sample("month_effect_country", dist.Normal(0.0, 0.3).expand([n_countries, n_months]))
    month_country = month_country - month_country.mean(axis=-1, keepdims=True)

    # non-centered reparameterization
    # (based on: https://mc-stan.org/docs/2_18/stan-users-guide/reparameterization-section.html)
    tau_region_season = numpyro.sample("tau_region_season", dist.HalfNormal(0.5))
    month_region_raw = numpyro.sample("month_region_raw",
                                      dist.Normal(0.0, 1.0).expand([n_regions, n_months]))
    month_region = month_region_raw * tau_region_season
    # Ensure zero-centering
    month_region = month_region - month_region.mean(axis=-1, keepdims=True)

    numpyro.deterministic("month_effect_region", month_region)

    log_lambda = (mu_country[country_idx] + region_effect +
                  month_country[country_idx, month_idx] +
                  month_region[region_idx, month_idx])

    rate = jnp.exp(jnp.clip(log_lambda, -20, 20)) * exposure
    numpyro.sample("obs", dist.Poisson(jnp.clip(rate, a_min=1e-10)), obs=accidents)


def run_mcmc_analysis(df, num_warmup=2000, num_samples=4000, num_chains=4):
    country_map = {"Germany": 0, "UK": 1}
    region_map = {rid: i for i, rid in enumerate(df["region_id"].unique())}

    model_inputs = {
        "accidents": df["accident_count"].to_numpy(),
        "exposure": (df["exposure"] / 1e9).to_numpy(),
        "country_idx": df["country"].map(country_map).to_numpy(),
        "region_idx": df["region_id"].map(region_map).to_numpy(),
        "month_idx": (df["month"] - 1).to_numpy(),  # 0-indexed for JAX
        "n_countries": 2,
        "n_regions": len(region_map)
    }

    mcmc = MCMC(NUTS(hierarchical_poisson_model), num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(jax.random.PRNGKey(0), **model_inputs)
    mcmc.print_summary()
    return mcmc.get_samples()


def compute_seasonal_rates(posterior, country_idx_per_region, seasons):
    mu = posterior["mu_country"]
    sigma = posterior["sigma_country"]
    region_raw = posterior["region_raw"]
    m_country = posterior["month_effect_country"]
    m_region = posterior["month_effect_region"]

    # Align parameters to regions
    mu_r = np.take(mu, country_idx_per_region, axis=1)
    sigma_r = np.take(sigma, country_idx_per_region, axis=1)

    results = {}
    for season, months in seasons.items():
        # Average seasonal effects across specified months
        m_c_eff = np.take(m_country, country_idx_per_region, axis=1)[:, :, months].mean(axis=-1)
        m_r_eff = m_region[:, :, months].mean(axis=-1)

        log_lambda = mu_r + (sigma_r * region_raw) + m_c_eff + m_r_eff
        results[season] = np.exp(log_lambda).mean(axis=0)
    return results


def plot_country_seasonality(posterior, path, country_names=("Germany", "UK")):
    plt.rcParams.update(bundles.icml2024(column="half", nrows=1, ncols=1))

    month_eff = posterior["month_effect_country"]
    months = np.arange(12)
    month_labels = [calendar.month_name[i + 1][:3] for i in months]
    colors = ["darkorange", "royalblue"]

    fig, ax = plt.subplots()

    ax.axhline(1, color="black", linestyle="--", alpha=0.6, label="Annual Average")

    for i, name in enumerate(country_names):
        mean = np.exp(month_eff[:, i, :].mean(axis=0))
        low, high = np.percentile(np.exp(month_eff[:, i, :]), [5, 95], axis=0)

        ax.plot(months, mean, label=name, color=colors[i])
        ax.fill_between(months, low, high, color=colors[i], alpha=0.2, lw=0)

    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.set_ylabel(r"Relative Risk $\exp(\gamma_{c[j],m})$")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(path)
    plt.show()


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

    proc = AccidentDataProcessor()

    # Germany
    ger_regions = gpd.read_file(BASE_DIR / "data/preprocessed/germany/geofiles/ger_gdf_with_osm_roads.gpkg")
    ger_acc = proc.load_accidents(BASE_DIR / "data/preprocessed/germany/collisions/preprocessed_ger.csv",
                                  category_filters={"casualty_severity": [1]})
    ger_merged = proc.aggregate_by_region_monthly(ger_regions, ger_acc)

    # UK
    uk_regions = gpd.read_file(BASE_DIR / "data/preprocessed/uk/geofiles/uk_gdf_with_osm_roads.gpkg")
    uk_acc = proc.load_accidents(BASE_DIR / "data/preprocessed/uk/collisions/preprocessed_uk.csv",
                                 category_filters={"casualty_severity": [1]})
    uk_merged = proc.aggregate_by_region_monthly(uk_regions, uk_acc)

    samples_path = BASE_DIR / "data/mcmc/mcmc_samples_region.npz"
    if not samples_path.exists():
        bayes_df = build_bayes_dataset(ger_merged, uk_merged)
        posterior = run_mcmc_analysis(bayes_df)
        np.savez(samples_path, **{k: np.array(v) for k, v in posterior.items()})
    else:
        data = np.load(samples_path)
        posterior = {k: data[k] for k in data.files}

    plot_path = BASE_DIR / 'results' / 'figures' /'seasonal_effect.pdf'
    plot_country_seasonality(posterior, plot_path)


if __name__ == "__main__":
    main()
