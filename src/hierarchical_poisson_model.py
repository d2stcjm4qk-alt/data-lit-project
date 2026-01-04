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


def build_bayes_dataset(germany_gdf, uk_gdf):
    """
    Combine Germany and UK region data into a single DataFrame
    with region_id, accidents, population, and country columns.
    """
    # Germany
    df_de = pd.DataFrame({
        "region_id": germany_gdf["region_code"].astype(str),
        "accidents": germany_gdf["accident_count"].astype(int),
        "population": germany_gdf["population"].fillna(0).astype(int),
        "country": "Germany"
    })

    # UK
    df_uk = pd.DataFrame({
        "region_id": uk_gdf["region_code"].astype(str),
        "accidents": uk_gdf["accident_count"].astype(int),
        "population": uk_gdf["population"].fillna(0).astype(int),
        "country": "UK"
    })

    df = pd.concat([df_de, df_uk], ignore_index=True)
    print(f"Merged DE+UK regions: {len(df)} rows")
    return df


# ==========================
# Hierarchical Poisson model
# ==========================
def hierarchical_poisson_model(accidents, population, country_idx, n_countries=2):
    n_regions = len(accidents)

    # Country-level hyperpriors
    mu_country = numpyro.sample("mu_country", dist.Normal(0, 5).expand([n_countries]))
    sigma_country = numpyro.sample("sigma_country", dist.HalfNormal(2).expand([n_countries]))

    # Region-level effects
    region_effect = numpyro.sample("region_effect", dist.Normal(0, 1).expand([n_regions]))

    # Log rate per 100k people
    log_lambda = mu_country[country_idx] + sigma_country[country_idx] * region_effect

    # Poisson likelihood scaled to 100k population
    numpyro.sample(
        "obs",
        dist.Poisson(jnp.exp(log_lambda) * (population / 1e5)),
        obs=accidents
    )


def hierarchical_poisson_model_aadf(accidents, exposure, country_idx, n_countries=2):
    n_regions = len(accidents)

    # Country-level log-rate priors (accidents per million vehicle-km)
    mu_country = numpyro.sample(
        "mu_country",
        dist.Normal(-10, 5).expand([n_countries])  # negative = rare events
    )

    sigma_country = numpyro.sample(
        "sigma_country",
        dist.HalfNormal(2).expand([n_countries])
    )

    # Region-level effects
    region_effect = numpyro.sample(
        "region_effect",
        dist.Normal(0, 1).expand([n_regions])
    )

    # Log accident rate
    log_lambda = mu_country[country_idx] + sigma_country[country_idx] * region_effect

    # NUMERICALLY SAFE RATE
    rate = jnp.exp(jnp.clip(log_lambda, -20, 20)) * exposure
    rate = jnp.clip(rate, a_min=1e-10)

    numpyro.sample(
        "obs",
        dist.Poisson(rate),
        obs=accidents
    )


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


# ==========================
# Extract posterior rates per 100k population
# ==========================
def extract_rates_per_100k(mcmc, df):
    """
    Convert MCMC samples to expected accident rates per 100k population per region.
    """
    samples = mcmc.get_samples()
    mu_country = samples['mu_country']  # (num_samples, n_countries)
    sigma_country = samples['sigma_country']
    region_effect = samples['region_effect']  # (num_samples, n_regions)

    country_idx = df['country'].map({'Germany': 0, 'UK': 1}).to_numpy()

    # Log per-person rate per sample
    log_lambda_samples = mu_country[:, country_idx] + sigma_country[:, country_idx] * region_effect.T
    lambda_samples = np.exp(log_lambda_samples)  # per-person rate

    # Convert to per 100k population
    lambda_100k = lambda_samples

    # Posterior mean and 95% HDI
    rate_mean = lambda_100k.mean(axis=0)
    rate_hdi = az.hdi(lambda_100k, hdi_prob=0.95)

    df_out = df.copy()
    df_out['rate_mean'] = rate_mean
    df_out['rate_lower'] = rate_hdi[:, 0]
    df_out['rate_upper'] = rate_hdi[:, 1]

    return df_out


# ==========================
# Plot expected rates
# ==========================
def plot_expected_accidents(df_probs, mcmc):
    import matplotlib.pyplot as plt
    import numpy as np
    import arviz as az

    fig, ax = plt.subplots(figsize=(16, 6))

    # ------------------- reconstruct posterior rate samples -------------------
    samples = mcmc.get_samples()
    country_idx = df_probs['country_idx'].values  # numeric 0/1
    population = df_probs['population'].values  # for exposure

    n_samples = samples['region_effect'].shape[0]
    n_regions = len(df_probs)

    # Reconstruct log per-person rates: log_lambda = mu_country + sigma_country * region_effect
    log_lambda_samples = samples["mu_country"][:, country_idx] + \
                         samples["sigma_country"][:, country_idx] * samples["region_effect"]

    # Convert to per-person rate
    lambda_samples = np.exp(log_lambda_samples)

    # Convert to per 100,000 population
    lambda_100k = lambda_samples

    # Compute posterior mean and 95% HDI per region
    rate_mean = lambda_100k.mean(axis=0)
    rate_hdi = az.hdi(lambda_100k, hdi_prob=0.95)

    df_probs["rate_mean"] = rate_mean
    df_probs["rate_lower"] = rate_hdi[:, 0]
    df_probs["rate_upper"] = rate_hdi[:, 1]

    # ------------------- plotting -------------------
    countries = df_probs['country'].unique()
    colors = {'Germany': '#4c72b0', 'UK': '#dd8452'}
    colors_mean = {'Germany': '#1e3457', 'UK': '#803b28'}

    # Plot region-level points with 95% CI
    for c in countries:
        df_c = df_probs[df_probs['country'] == c]
        idx = np.arange(len(df_c))
        cm = df_c["rate_mean"].values
        cl = df_c["rate_lower"].values
        cu = df_c["rate_upper"].values
        color = colors[c]

        # Vertical CI lines
        for i, m, low, up in zip(idx, cm, cl, cu):
            ax.vlines(i, low, up, color=color, alpha=0.6, lw=1)
            cap = 0.9
            ax.hlines([low, up], i - cap, i + cap, color=color, lw=1)

        # Mean points
        ax.plot(idx, cm, "o", color=color, markersize=5, alpha=0.5, label=f"{c} regions")

    # Plot country-level mean and 95% HDI tube
    for c in countries:
        df_c = df_probs[df_probs['country'] == c]
        region_indices = df_c.index.values

        country_draw_means = lambda_100k[:, region_indices].mean(axis=1)

        c_mean = country_draw_means.mean()
        c_hdi = az.hdi(country_draw_means, hdi_prob=0.95)

        color = colors_mean[c]
        x_min, x_max = 0, 171

        # Shaded HDI tube
        ax.fill_between([x_min, x_max], [c_hdi[0], c_hdi[0]], [c_hdi[1], c_hdi[1]],
                        color=color, alpha=0.12)

        # Mean line
        ax.hlines(c_mean, x_min, x_max, color=color, linestyle="--", lw=2,
                  label=f"{c} mean")

    # ------------------- styling -------------------
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlabel("Region index")
    ax.set_ylabel("Expected accident rate per 100,000 people")
    ax.set_title("Posterior expected accident rate per region (with country means)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_expected_accidents_aadf(df_probs, mcmc):
    """
    Plot posterior expected accident rates per region (per million vehicle-km)
    using NumPy arrays from MCMC samples. Top 10% of German regions are removed
    for plotting.

    Parameters:
    -----------
    df_probs : pd.DataFrame
        DataFrame containing region info, 'country', 'country_idx', 'exposure' columns
    samples : dict
        Dictionary with NumPy arrays from MCMC: 'mu_country', 'sigma_country', 'region_effect'
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import arviz as az

    fig, ax = plt.subplots(figsize=(16, 6))

    country_idx = df_probs['country_idx'].values
    exposure_million = df_probs['exposure'].values / 1e6  # per million vehicle-km

    # ------------------- reconstruct posterior rate samples -------------------
    # shapes:
    # mu_country: (n_samples, n_countries)
    # sigma_country: (n_samples, n_countries)
    # region_effect: (n_samples, n_regions)
    samples = mcmc.get_samples()
    mu = samples['mu_country']
    sigma = samples['sigma_country']
    region_effect = samples['region_effect']

    n_samples = region_effect.shape[0]
    n_regions = region_effect.shape[1]

    # Compute rate per region per sample
    log_lambda = mu[:, country_idx] + sigma[:, country_idx] * region_effect  # shape (n_samples, n_regions)
    lambda_per_million = np.exp(log_lambda)  # accidents per million vehicle-km

    # Multiply by exposure if you want expected counts (optional)
    # expected_counts = lambda_per_million * exposure_million

    # ------------------- region-level posterior summaries -------------------
    rate_mean = lambda_per_million.mean(axis=0)
    rate_hdi = az.hdi(lambda_per_million, hdi_prob=0.95)

    df_probs = df_probs.copy()
    df_probs['rate_mean'] = rate_mean
    df_probs['rate_lower'] = rate_hdi[:, 0]
    df_probs['rate_upper'] = rate_hdi[:, 1]
    print('ääääääääääääääääääää')
    print(df_probs['country'])

    # ------------------- remove top 10% German regions -------------------
    mask_de = df_probs['country'] == 'Germany'
    threshold = df_probs.loc[mask_de, 'rate_mean'].quantile(0.90)
    df_plot = df_probs[~((df_probs['country'] == 'Germany') & (df_probs['rate_mean'] > threshold))].reset_index(
        drop=True)
    idx_plot = df_plot.index.values

    # ------------------- plotting -------------------
    countries = df_plot['country'].unique()
    colors = {'Germany': '#4c72b0', 'UK': '#dd8452'}
    colors_mean = {'Germany': '#1e3457', 'UK': '#803b28'}

    # Sort regions within country for nicer plotting
    df_plot = df_plot.sort_values(['country', 'rate_mean']).reset_index(drop=True)

    for c in countries:
        df_c = df_plot[df_plot['country'] == c]
        idx = np.arange(len(df_c))

        cm = df_c['rate_mean'].values
        cl = df_c['rate_lower'].values
        cu = df_c['rate_upper'].values
        color = colors[c]

        # Vertical CI lines
        for i, m, low, up in zip(idx, cm, cl, cu):
            ax.vlines(i, low, up, color=color, alpha=0.6, lw=1)
            cap = 0.25
            ax.hlines([low, up], i - cap, i + cap, color=color, lw=1)

        # Mean points
        ax.plot(idx, cm, 'o', color=color, markersize=5, alpha=0.6, label=f'{c} regions')
        '''
        # Label TOP 5 highest-risk regions per country
        df_top5 = df_c.nlargest(5, 'rate_mean')
        for _, row in df_top5.iterrows():
            i = df_c.index.get_loc(row.name)
            ax.text(i, row['rate_mean'], str(row.get('region_id', '')),
                    fontsize=9, ha='left', va='bottom', rotation=30, color=color, alpha=0.9)
        '''

    # ------------------- country-level mean & HDI tubes -------------------
    for c in countries:
        df_c = df_plot[df_plot['country'] == c]
        region_indices = df_c.index.values

        country_draw_means = lambda_per_million[:, region_indices].mean(axis=1)

        c_mean = country_draw_means.mean()
        c_hdi = az.hdi(country_draw_means, hdi_prob=0.95)

        color = colors_mean[c]
        x_min, x_max = 0, 171 - int(171 * 0.1)

        ax.fill_between([x_min, x_max],
                        [c_hdi[0], c_hdi[0]],
                        [c_hdi[1], c_hdi[1]],
                        color=color, alpha=0.12)
        ax.hlines(c_mean, x_min, x_max, color=color, linestyle='--', lw=2, label=f'{c} mean')

    # ------------------- styling -------------------
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.set_xlabel("Region index (sorted within country)")
    ax.set_ylabel("Accidents per Million Vehicle-Kilometers")
    ax.set_title("Posterior Expected Accident Risk per Region\n(with Country-Level Means & 95% HDI)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return df_plot


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


def compute_country_posteriors(mcmc, df):
    """
    Compute the posterior distribution of accident rates per country
    (population-weighted mean accident rate per 100k people).
    """

    import numpy as np
    import arviz as az

    samples = mcmc.get_samples()

    mu_country = samples['mu_country']  # shape: (S, 2)
    sigma_country = samples['sigma_country']  # shape: (S, 2)
    region_effect = samples['region_effect']  # shape: (S, R)

    # region-level country index and population
    country_idx = df["country"].map({"Germany": 0, "UK": 1}).to_numpy()
    population = df["population"].to_numpy()

    # reconstruct log-rate per region (S x R)
    log_lambda_samples = (
            mu_country[:, country_idx] +
            sigma_country[:, country_idx] * region_effect
    )

    # per-person rate, scaled to per 100k
    lambda_100k = np.exp(log_lambda_samples)  # already per 100k population

    # ----- compute population-weighted national rate per sample -----
    country_results = {}
    for country_name, idx in {"Germany": 0, "UK": 1}.items():
        # select region indices belonging to this country
        mask = (country_idx == idx)
        pop_c = population[mask]
        rate_c = lambda_100k[:, mask]  # shape S x num_regions_in_country

        # weighted national rate for each posterior sample:
        weighted_country_rate = (rate_c * pop_c).sum(axis=1) / pop_c.sum()

        # compute posterior summary
        mean_rate = weighted_country_rate.mean()
        hdi = az.hdi(weighted_country_rate, hdi_prob=0.95)

        country_results[country_name] = {
            "posterior_samples": weighted_country_rate,
            "mean": mean_rate,
            "hdi_lower": hdi[0],
            "hdi_upper": hdi[1],
        }

    return country_results


def compute_country_posteriors_aadf(mcmc, df):
    """
    Compute posterior distribution of national accident rates
    (EXPOSURE-weighted accident rate per million vehicle-km).
    """

    import numpy as np
    import arviz as az

    samples = mcmc.get_samples()

    mu_country = samples['mu_country']  # shape: (S, 2)
    sigma_country = samples['sigma_country']  # shape: (S, 2)
    region_effect = samples['region_effect']  # shape: (S, R)

    # region-level country index and exposure
    country_idx = df["country"].map({"Germany": 0, "UK": 1}).to_numpy()

    # per million vehicle-km
    df["exposure"] = df["exposure"] / 1e6
    exposure = df["exposure"].to_numpy()

    # reconstruct log-rate per region (S x R)
    log_lambda_samples = (
            mu_country[:, country_idx] +
            sigma_country[:, country_idx] * region_effect
    )

    # accident rate per million vehicle-km
    lambda_per_million_vkm = np.exp(log_lambda_samples)

    # ----- compute exposure-weighted national rate per sample -----
    country_results = {}

    for country_name, idx in {"Germany": 0, "UK": 1}.items():
        mask = (country_idx == idx)

        exposure_c = exposure[mask]  # (R_c,)
        rate_c = lambda_per_million_vkm[:, mask]  # (S, R_c)

        # exposure-weighted national rate
        weighted_country_rate = (
                (rate_c * exposure_c).sum(axis=1) / exposure_c.sum()
        )

        mean_rate = weighted_country_rate.mean()
        hdi = az.hdi(weighted_country_rate, hdi_prob=0.95)

        country_results[country_name] = {
            "posterior_samples": weighted_country_rate,
            "mean": mean_rate,
            "hdi_lower": hdi[0],
            "hdi_upper": hdi[1],
        }

    return country_results


def plot_country_posteriors(country_posteriors):
    """
    Plot posterior density curves for each country's accident rate
    (accidents per 100k population).

    Input:
        country_posteriors = output of compute_country_posteriors()
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 6))

    colors = {
        "Germany": "#365f9c",
        "UK": "#d97a33"
    }

    for country, result in country_posteriors.items():
        samples = result["posterior_samples"]
        mean = result["mean"]
        low = result["hdi_lower"]
        high = result["hdi_upper"]

        # KDE density line
        sns.kdeplot(samples, label=f"{country} posterior", linewidth=2, color=colors[country])

        # Mean vertical line
        plt.axvline(mean, color=colors[country], linestyle="--", linewidth=2)

        # 95% credible interval shading
        plt.fill_betweenx(
            [0, plt.gca().get_ylim()[1]],
            low, high,
            color=colors[country],
            alpha=0.15,
            label=f"{country} 95% CI"
        )

    plt.title("Posterior Accident Rate Distribution per Country (per 100k people)", fontsize=15)
    plt.xlabel("Accidents per 100,000 population", fontsize=13)
    plt.ylabel("Density", fontsize=13)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_country_posteriors_aadf(country_posteriors, df_probs):
    """
    Plot posterior density curves for each country's accident rate
    (accidents per million vehicle-kilometers), excluding the top 10%
    of German regions if requested.

    df_probs: DataFrame containing 'country' and 'rate_mean' per region.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    plt.figure(figsize=(12, 6))

    colors = {
        "Germany": "#365f9c",
        "UK": "#d97a33"
    }

    for country, result in country_posteriors.items():
        samples = result["posterior_samples"]

        # Exclude top 10% German regions
        if country == "Germany":
            mask_de = df_probs["country"] == "Germany"
            threshold = df_probs.loc[mask_de, "rate_mean"].quantile(0.90)

            # Keep only samples from regions below threshold
            keep_regions = df_probs.loc[mask_de & (df_probs["rate_mean"] <= threshold)].index
            # If posterior_samples is 1D, we just filter using region IDs
            samples = np.array([s for i, s in enumerate(samples) if i in keep_regions])

        mean = samples.mean()  # recompute mean after filtering

        # KDE plot
        sns.kdeplot(
            samples,
            label=f"{country} posterior",
            linewidth=2,
            color=colors[country]
        )

        # Mean line
        plt.axvline(mean, color=colors[country], linestyle="--", linewidth=2)

    plt.title(
        "Posterior Accident Risk per Country\n(Accidents per Million Vehicle-Kilometers)",
        fontsize=15
    )
    plt.xlabel("Accidents per Million Vehicle-Kilometers", fontsize=13)
    plt.ylabel("Density", fontsize=13)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


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
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import contextily as ctx
    import numpy as np
    from mpl_toolkits.axes_grid1 import make_axes_locatable

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
