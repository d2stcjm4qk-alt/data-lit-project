import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import contextily as ctx
from typing import Optional, Dict, Tuple, List, Union
import os
from shapely.ops import unary_union
from libpysal import weights
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Load accidents data
def load_accidents(
        csv_path: Union[str, Path],
        lon_col: str = "longitude",
        lat_col: str = "latitude",
        separator: str = ",",
        category_filters: Optional[Dict[str, List]] = None,
        target_crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Load accident CSV and convert to GeoDataFrame.
    Optionally filter by multiple column values (e.g., casualty_severity, IstKrad).
    Ensures consistent CRS for spatial operations.

    Parameters:
        csv_path (str or Path): Path to the CSV file.
        lon_col (str): Longitude column name.
        lat_col (str): Latitude column name.
        separator (str): CSV delimiter.
        category_filters (dict, optional): Dictionary of {column: [values]} to filter.
        target_crs (str): CRS to set for the GeoDataFrame (default "EPSG:4326").

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with filtered accidents and geometry column.
    """
    # Load CSV
    df = pd.read_csv(str(csv_path), sep=separator, low_memory=False)

    # Convert longitude and latitude to floats
    df[lon_col] = df[lon_col].astype(str).str.replace(",", ".").astype(float)
    df[lat_col] = df[lat_col].astype(str).str.replace(",", ".").astype(float)

    # Apply category filters if provided
    if category_filters:
        for col, values in category_filters.items():
            if col in df.columns:
                df = df[df[col].isin(values)]
            else:
                print(f"Warning: Column '{col}' not found in CSV.")

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"  # assume lon/lat in WGS84
    )

    # Reproject to target CRS if needed
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    print(f"Loaded {len(gdf):,} accident points from {csv_path} (CRS={gdf.crs})")
    return gdf


def add_accident_counts_to_regions_monthly(
        regions_gdf: gpd.GeoDataFrame,
        accidents_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Adds accident counts per region per month (1–12).
    Returns one row per (region, month).
    """

    regions_gdf = regions_gdf.to_crs(accidents_gdf.crs)

    # Extract month
    accidents_gdf = accidents_gdf.copy()

    # Spatial join
    accidents_with_region = gpd.sjoin(
        accidents_gdf,
        regions_gdf,
        how="left",
        predicate="within"
    )

    # Count accidents per region × month
    counts = (
        accidents_with_region
        .groupby(["index_right", "month"])
        .size()
        .rename("accident_count")
        .reset_index()
    )

    # Create full grid: region × month
    full_index = pd.MultiIndex.from_product(
        [regions_gdf.index, range(1, 13)],
        names=["index_right", "month"]
    ).to_frame(index=False)

    counts_full = full_index.merge(
        counts,
        on=["index_right", "month"],
        how="left"
    )

    counts_full["accident_count"] = counts_full["accident_count"].fillna(0).astype(int)

    # Merge geometry back
    result = counts_full.merge(
        regions_gdf,
        left_on="index_right",
        right_index=True,
        how="left"
    )

    return gpd.GeoDataFrame(result, geometry="geometry", crs=regions_gdf.crs)


def build_traffic_volume_bayes_dataset_seasonal(germany_gdf, uk_gdf):
    """
    Build dataset for hierarchical Poisson model with seasonal (monthly) effects.
    Exposure is annual vehicle-km divided equally across months.
    """

    def process_gdf(gdf, country_name):
        df = gdf.copy()
        print(f"{country_name} columns: {df.columns}")

        # Compute AADF weighted by road length
        df["AADF_region_combined"] = (
                (df["AADF_A"].fillna(0) * df["osm_length_A_km"].fillna(0) +
                 df["AADF_B"].fillna(0) * df["osm_length_B_km"].fillna(0))
                / df["osm_total_length_km"].replace(0, np.nan)
        )

        # Annual vehicle-km exposure
        df["exposure_annual"] = (
                df["AADF_region_combined"] * df["osm_total_length_km"]
        )

        df["exposure_annual"] = (
            df["exposure_annual"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .clip(lower=0)
        )

        # Monthly exposure (assumed uniform)
        df["exposure"] = df["exposure_annual"] / 12.0

        return pd.DataFrame({
            "region_id": df["region_code"].astype(str),
            "month": df["month"].astype(int),  # 1–12
            "accident_count": df["accident_count"].astype(int),
            "exposure": df["exposure"],

            # Diagnostics / controls
            "population": df["population"].fillna(0).astype(int),
            "AADF_region_combined": df["AADF_region_combined"],
            "osm_total_length_km": df["osm_total_length_km"],

            "country": country_name
        })

    df_de = process_gdf(germany_gdf, "Germany")
    df_uk = process_gdf(uk_gdf, "UK")

    df_combined = pd.concat([df_de, df_uk], ignore_index=True)
    print(f"Merged DE+UK seasonal rows: {len(df_combined)}")

    return df_combined


def hierarchical_poisson_model_region_season(
        accidents,
        exposure,
        country_idx,
        region_idx,
        month_idx,
        n_countries,
        n_regions,
        n_months=12
):
    # ----------------------------
    # Country-level baseline log rates
    # ----------------------------
    mu_country = numpyro.sample(
        "mu_country",
        dist.Normal(-10.0, 5.0).expand([n_countries])
    )

    # ----------------------------
    # Country-level regional heterogeneity
    # ----------------------------
    sigma_country = numpyro.sample(
        "sigma_country",
        dist.HalfNormal(1.5).expand([n_countries])
    )

    # ----------------------------
    # Region random intercepts (shared scale, country-modulated)
    # ----------------------------
    region_raw = numpyro.sample(
        "region_raw",
        dist.Normal(0.0, 1.0).expand([n_regions])
    )

    region_effect = sigma_country[country_idx] * region_raw[region_idx]

    # ----------------------------
    # Country-level seasonal pattern
    # ----------------------------
    month_effect_country = numpyro.sample(
        "month_effect_country",
        dist.Normal(0.0, 0.3).expand([n_countries, n_months])
    )

    month_effect_country = (
            month_effect_country
            - month_effect_country.mean(axis=-1, keepdims=True)
    )

    # ----------------------------
    # Regional seasonal deviations
    # ----------------------------
    tau_region_season = numpyro.sample(
        "tau_region_season",
        dist.HalfNormal(0.15)
    )

    month_effect_region = numpyro.sample(
        "month_effect_region",
        dist.Normal(0.0, tau_region_season).expand([n_regions, n_months])
    )

    month_effect_region = (
            month_effect_region
            - month_effect_region.mean(axis=-1, keepdims=True)
    )

    # ----------------------------
    # Latent regional log-rate
    # (this is what you want!)
    # ----------------------------
    log_lambda_region = (
            mu_country[country_idx]
            + region_effect
            + month_effect_country[country_idx, month_idx]
            + month_effect_region[region_idx, month_idx]
    )

    # ----------------------------
    # Poisson likelihood
    # ----------------------------
    rate = jnp.exp(jnp.clip(log_lambda_region, -20, 20)) * exposure
    rate = jnp.clip(rate, a_min=1e-10)

    numpyro.sample("obs", dist.Poisson(rate), obs=accidents)


# ==========================
# Run NumPyro MCMC
# ==========================
def run_numpyro_model(df, num_warmup=2000, num_samples=4000, num_chains=5):
    """
    Run MCMC for the hierarchical Poisson model.
    """
    accidents = df["accidents"].to_numpy()
    population = df["population"].to_numpy()

    # Map countries to integer indices
    df['country_idx'] = df['country'].map({'Germany': 0, 'UK': 1}).astype(int)
    country_idx = df['country_idx'].to_numpy()

    kernel = NUTS(hierarchical_poisson_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(jax.random.PRNGKey(0),
             accidents=accidents,
             population=population,
             country_idx=country_idx)
    mcmc.print_summary()
    return mcmc


def run_numpyro_model_aadf(df, num_warmup=2000, num_samples=4000, num_chains=5):
    """
    Run MCMC for the hierarchical Poisson model using vehicle-km exposure.
    """

    accidents = df["accidents"].to_numpy()
    df["exposure"] = df["exposure"] / 1e6  # per million vehicle-km
    exposure = df["exposure"].to_numpy()

    # Map countries to integer indices
    df['country_idx'] = df['country'].map({'Germany': 0, 'UK': 1}).astype(int)
    country_idx = df['country_idx'].to_numpy()

    kernel = NUTS(hierarchical_poisson_model_aadf)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains
    )

    mcmc.run(
        jax.random.PRNGKey(0),
        accidents=accidents,
        exposure=exposure,
        country_idx=country_idx
    )

    mcmc.print_summary()
    return mcmc


def run_numpyro_model_aadf_seasonal(
        df,
        num_warmup=2000,
        num_samples=4000,
        num_chains=4
):
    """
    Run hierarchical Poisson model with country-specific seasonal (monthly) effects.
    """

    # Observations
    print(f'run moden df: {df.columns}')
    print(f'MCMC df: {df}')
    accidents = df["accident_count"].to_numpy()

    # Exposure in million vehicle-km (monthly exposure)
    exposure = (df["exposure"] / 1e6).to_numpy()

    # Country index
    country_idx = df["country"].map(
        {"Germany": 0, "UK": 1}
    ).astype(int).to_numpy()

    # Region index (shared across months)
    region_mapping = {rid: i for i, rid in enumerate(df["region_id"].unique())}
    df["region_idx"] = df["region_id"].map(region_mapping)
    region_idx = df["region_idx"].astype(int).to_numpy()

    # Month index: must be 0–11
    month_idx = df["month"].astype(int).to_numpy()

    n_countries = 2
    n_regions = len(region_mapping)
    n_months = 12

    kernel = NUTS(hierarchical_poisson_model_region_season)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains
    )

    mcmc.run(
        jax.random.PRNGKey(0),
        accidents=accidents,
        exposure=exposure,
        country_idx=country_idx,
        region_idx=region_idx,
        month_idx=month_idx,
        n_countries=n_countries,
        n_regions=n_regions,
        n_months=n_months
    )

    mcmc.print_summary()
    return mcmc


# --- Posterior predictive for region probabilities ---
def get_region_probs(mcmc, df):
    samples = mcmc.get_samples()

    # country indices
    country_idx = df['country'].map({'Germany': 0, 'UK': 1}).to_numpy()

    # Extract samples
    mu_country = samples['mu_country']  # shape: (num_samples, n_countries)
    region_effect = samples['region_effect']  # shape: (num_samples, n_regions)

    # Compute log accident rates per region
    log_rate_samples = mu_country[:, country_idx] + region_effect

    # Convert to rates
    rate_samples = np.exp(log_rate_samples)

    # Compute posterior mean and 95% HDI per region
    rate_mean = rate_samples.mean(axis=0)
    import arviz as az
    rate_hdi = az.hdi(rate_samples, hdi_prob=0.95)

    # Combine into DataFrame
    df_probs = df.copy()
    df_probs['rate_mean'] = rate_mean
    df_probs['rate_lower'] = rate_hdi[:, 0]
    df_probs['rate_upper'] = rate_hdi[:, 1]

    return df_probs


def plot_seasonality_by_country(samples):
    """
    Plot posterior mean and 95% credible interval of monthly relative risk
    for Germany and the UK (country-level).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import arviz as az

    month_eff = samples["month_effect_country"]  # (samples, country, 12)
    sigma = samples["sigma_season"]  # (samples, country)

    months = np.arange(1, 13)
    countries = ["Germany", "UK"]
    colors = {"Germany": "#4c72b0", "UK": "#dd8452"}
    colors_mean = {"Germany": "#1e3457", "UK": "#803b28"}

    fig, ax = plt.subplots(figsize=(14, 6))

    for c_idx, country in enumerate(countries):
        # --- compute seasonal log-effect ---
        seasonal_log = sigma[:, c_idx, None] * month_eff[:, c_idx, :]

        # --- re-center per posterior draw (CRUCIAL) ---
        seasonal_log = seasonal_log - seasonal_log.mean(axis=1, keepdims=True)

        # --- relative risk ---
        rr_samples = np.exp(seasonal_log)

        rr_mean = rr_samples.mean(axis=0)
        rr_hdi = az.hdi(rr_samples, hdi_prob=0.95)

        ax.fill_between(
            months,
            rr_hdi[:, 0],
            rr_hdi[:, 1],
            color=colors[country],
            alpha=0.25
        )

        ax.plot(
            months,
            rr_mean,
            marker="o",
            lw=2.5,
            color=colors_mean[country],
            label=country
        )

    ax.axhline(1.0, color="black", linestyle="--", lw=1)
    ax.set_xticks(months)
    ax.set_xlabel("Month")
    ax.set_ylabel("Relative Accident Risk")
    ax.set_title(
        "Seasonal Variation in Fatal Road Accident Risk\n"
        "(Posterior Mean and 95% Credible Interval)"
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.show()


def compute_seasonal_region_rates_from_posterior(
        posterior,
        country_idx_per_region,
        seasons,
        reduce="mean"
):
    """
    Returns:
        seasonal_rates: dict(season -> array[n_regions])
    """

    mu = posterior["mu_country"]  # (S, C)
    sigma = posterior["sigma_country"]  # (S, C)
    region_raw = posterior["region_raw"]  # (S, R)
    month_country = posterior["month_effect_country"]  # (S, C, M)
    month_region = posterior["month_effect_region"]  # (S, R, M)

    S, R = region_raw.shape
    seasonal_rates = {}

    # 🔑 correct per-region country parameters
    mu_region = np.take(mu, country_idx_per_region, axis=1)  # (S, R)
    sigma_region = np.take(sigma, country_idx_per_region, axis=1)  # (S, R)
    print(f'sigma_region shape: {sigma_region.shape}')

    for season, months in seasons.items():
        log_lambda = (
                mu_region
                + sigma_region * region_raw
                + np.take(month_country, country_idx_per_region, axis=1)[:, :, months].mean(axis=-1)
                + month_region[:, :, months].mean(axis=-1)
        )  # (S, R)

        lambda_rate = np.exp(log_lambda)

        if reduce == "mean":
            seasonal_rates[season] = lambda_rate.mean(axis=0)
        elif reduce == "median":
            seasonal_rates[season] = np.median(lambda_rate, axis=0)
        else:
            raise ValueError("reduce must be 'mean' or 'median'")

    return seasonal_rates


def plot_dominant_season_country(
        regions_gdf,
        seasonal_rates,
        country_name,
        save_path=None
):
    gdf = regions_gdf.to_crs(epsg=3857)

    season_names = list(seasonal_rates.keys())
    values = np.vstack([seasonal_rates[s] for s in season_names]).T

    dominant_idx = np.argmax(values, axis=1)
    gdf["dominant_season"] = np.array(season_names)[dominant_idx]
    gdf["dominant_rate"] = values[np.arange(len(gdf)), dominant_idx]

    # Define seasonal colormaps
    season_cmaps = {
        "Winter": "Blues",
        "Spring": "Greens",
        "Summer": "YlOrBr",
        "Autumn": "Greys",
    }

    # Global norm for all seasons, avoid pure white for low values by setting vmin > 0.7
    global_vmin = 0.5
    global_vmax = 2.0
    norms = {s: mcolors.LogNorm(vmin=global_vmin, vmax=global_vmax) for s in season_names}

    # Plot main map
    fig, ax = plt.subplots(figsize=(8, 10))
    for season in season_names:
        mask = gdf["dominant_season"] == season
        if not mask.any():
            continue
        gdf.loc[mask].plot(
            ax=ax,
            column="dominant_rate",
            cmap=season_cmaps[season],
            norm=norms[season],
            edgecolor="gray",
            linewidth=0.3,
            alpha=0.85
        )

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=7, alpha=0.5)
    ax.set_title(f"Dominant Seasonal Accident Rate – {country_name}")
    ax.axis("off")

    # Create 2x2 grid of colorbars
    fig.subplots_adjust(right=0.8, hspace=0.4, wspace=0.4)
    cb_axs = []
    for i, season in enumerate(season_names):
        row = i // 2
        col = i % 2
        cb_ax = fig.add_axes([0.82 + col * 0.08, 0.65 - row * 0.25, 0.02, 0.18])
        sm = plt.cm.ScalarMappable(cmap=season_cmaps[season], norm=norms[season])
        sm._A = []
        cb = plt.colorbar(sm, cax=cb_ax)
        cb.ax.set_title(season, fontsize=9, pad=2)
        cb_axs.append(cb_ax)

    # Shared x-axis label beneath all colorbars
    fig.text(0.82, 0.05, "accidents per billion vehicle-km", rotation=0, fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def main():
    # Base directory (project root, one level up from src)
    BASE_DIR = Path(__file__).resolve().parent.parent

    # germany_geo_path = BASE_DIR / "data" / "processed" / "geo_data" / "Germany_merged.geojson"
    germany_geo_path = BASE_DIR / "data" / "preprocessed" / "germany" / "traffic" / "ger_gdf_with_osm_roads.gpkg"
    germany_acc_path = BASE_DIR / "data" / "processed" / "reduced_uk_dataset" / "modified_ger.csv"
    ger_regions_gdf = gpd.read_file(germany_geo_path)

    ger_accidents_gdf = load_accidents(
        str(germany_acc_path),
        category_filters={
            "casualty_severity": [1],
        }
    )

    ger_regions_with_accidents = add_accident_counts_to_regions_monthly(
        regions_gdf=ger_regions_gdf,
        accidents_gdf=ger_accidents_gdf
    )

    print(f'ger regions: {ger_regions_with_accidents}')

    # UK datasets
    # uk_geo_path = BASE_DIR / "data" / "processed" / "geo_data" / "UK_merged.geojson"
    uk_geo_path = BASE_DIR / "data" / "preprocessed" / "uk" / "traffic" / "uk_gdf_with_osm_roads.gpkg"
    # uk_acc_path = BASE_DIR / "data" / "processed" / "reduced_uk_dataset" / "reduced_uk_dataset.csv"
    uk_acc_path = BASE_DIR / "data" / "processed" / "reduced_uk_dataset" / "reduced_uk_dataset.csv"
    os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"  # No size limit
    uk_regions_gdf = gpd.read_file(uk_geo_path)

    uk_accidents_gdf = load_accidents(
        str(uk_acc_path),
        category_filters={
            "collision_severity": [1],
        }
    )
    # print("UK accidents CRS:", uk_accidents_gdf.crs)
    # print("UK regions CRS:", uk_regions_gdf.crs)
    # extract hour as integer

    uk_regions_with_accidents = add_accident_counts_to_regions_monthly(
        regions_gdf=uk_regions_gdf,
        accidents_gdf=uk_accidents_gdf
    )

    # print(uk_regions_with_accidents['accident_count'].describe())
    # print(uk_regions_with_accidents['accident_count'].value_counts().head())

    samples_path = BASE_DIR / "data" / "mcmc" / "mcmc_samples_region.npz"

    if not samples_path.exists():
        bayes_df = build_traffic_volume_bayes_dataset_seasonal(ger_regions_with_accidents, uk_regions_with_accidents)

        mcmc = run_numpyro_model_aadf_seasonal(bayes_df)

        posterior = mcmc.get_samples()
        np.savez(
            str(samples_path),
            **{k: np.array(v) for k, v in posterior.items()}
        )
    else:
        data = np.load(str(samples_path))
        posterior = {k: data[k] for k in data.files}

    print(f'posterior: {posterior}')
    # month_effect_country: (samples, country, month)
    month_eff = posterior["month_effect_country"]

    # Seasonality strength per sample & country
    # (std over months on log-rate scale)
    sigma_season = month_eff.std(axis=-1)  # shape: (samples, country)

    sigma_de = sigma_season[:, 0]
    sigma_uk = sigma_season[:, 1]

    print("Germany seasonality (std over months):")
    print(np.percentile(sigma_de, [5, 50, 95]))

    print("\nUK seasonality (std over months):")
    print(np.percentile(sigma_uk, [5, 50, 95]))

    # Probability Germany is more seasonal than UK
    prob_de_more = np.mean(sigma_de > sigma_uk)
    print(f"\nP(Germany more seasonal than UK) = {prob_de_more:.3f}")

    print("\nsigma_season summary:")
    print(
        sigma_season.min(),
        sigma_season.mean(),
        sigma_season.max()
    )

    print("\nmonth_effect_country summary:")
    print(
        month_eff.min(),
        month_eff.mean(),
        month_eff.max()
    )

    # month_effect_country: shape (samples, country, month)
    month_eff = posterior["month_effect_country"]

    months = np.arange(1, 13)

    # Compute posterior mean and 5th/95th percentiles
    mean_de = month_eff[:, 0, :].mean(axis=0)
    ci_de = np.percentile(month_eff[:, 0, :], [5, 95], axis=0)

    mean_uk = month_eff[:, 1, :].mean(axis=0)
    ci_uk = np.percentile(month_eff[:, 1, :], [5, 95], axis=0)

    plt.figure(figsize=(8, 4))

    # Germany
    plt.plot(months, mean_de, label="Germany", color="blue")
    plt.fill_between(months, ci_de[0], ci_de[1], color="blue", alpha=0.3)

    # UK
    plt.plot(months, mean_uk, label="UK", color="green")
    plt.fill_between(months, ci_uk[0], ci_uk[1], color="green", alpha=0.3)

    plt.xticks(months)
    plt.xlabel("Month")
    plt.ylabel("Country-level seasonal effect (log-rate)")
    plt.title("Posterior monthly seasonality by country")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    SEASONS = {
        "Winter": [11, 0, 1],
        "Spring": [2, 3, 4],
        "Summer": [5, 6, 7],
        "Autumn": [8, 9, 10],
    }
    ger_region_ids = (
        ger_regions_with_accidents["index_right"]
        .drop_duplicates()
        .sort_values()
        .to_numpy()
    )

    uk_region_ids = (
        uk_regions_with_accidents["index_right"]
        .drop_duplicates()
        .sort_values()
        .to_numpy()
    )
    country_idx_per_region = np.concatenate([
        np.zeros(len(ger_region_ids), dtype=int),  # Germany = 0
        np.ones(len(uk_region_ids), dtype=int)  # UK = 1
    ])

    print('111111111111111111111')
    seasonal_rates = compute_seasonal_region_rates_from_posterior(
        posterior=posterior,
        country_idx_per_region=country_idx_per_region,
        seasons=SEASONS,
        reduce="mean"  # or "median"
    )

    for season, rates in seasonal_rates.items():
        print(season, rates.min(), rates.mean(), rates.max())

    regions_gdf_all = pd.concat(
        [ger_regions_with_accidents, uk_regions_with_accidents],
        ignore_index=True
    )
    print(f'uk length: {len(uk_regions_with_accidents)}')

    regions_gdf_all = (
        regions_gdf_all
        .drop_duplicates(subset="index_right")
        .sort_values("index_right")
        .reset_index(drop=True)
    )

    # Must align with posterior region dimension
    print(f'regipons_gdf_all: {len(regions_gdf_all)}, country_idx_per_region: {len(country_idx_per_region)}')

    COUNTRY_MAP = {
        "Germany": 0,
        "UK": 1,
    }

    country_name = "Germany"  # or "UK"
    country_code = COUNTRY_MAP[country_name]

    mask = country_idx_per_region == country_code
    print(f'regions_gdf_all: {len(regions_gdf_all)}')
    regions_gdf_country = regions_gdf_all.copy()

    seasonal_rates_country = {
        season: rates[mask]
        for season, rates in seasonal_rates.items()
    }

    plot_dominant_season_country(
        regions_gdf=regions_gdf_country,
        seasonal_rates=seasonal_rates_country,
        country_name=country_name
    )


if __name__ == "__main__":
    main()
