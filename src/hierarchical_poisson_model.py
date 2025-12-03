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


def add_accident_counts_to_regions(
        regions_gdf: gpd.GeoDataFrame,
        accidents_gdf: gpd.GeoDataFrame,
        region_id_col: str = None
) -> gpd.GeoDataFrame:
    """
    Adds a column to regions_gdf with the count of accidents falling within each region.

    Parameters:
        regions_gdf (GeoDataFrame): GeoDataFrame with polygon geometries (regions).
        accidents_gdf (GeoDataFrame): GeoDataFrame with point geometries (accidents).
        region_id_col (str, optional): Column in regions_gdf to use as identifier (default uses index).

    Returns:
        GeoDataFrame: regions_gdf with a new column 'accident_count'.
    """

    regions_gdf = regions_gdf.to_crs(accidents_gdf.crs)

    # Spatial join: attach region info to each accident
    accidents_with_region = gpd.sjoin(
        accidents_gdf, regions_gdf, how="left", predicate="within"
    )

    # Count accidents per region
    accident_counts = (
        accidents_with_region
        .groupby("index_right")
        .size()
        .rename("accident_count")
    )

    # Merge counts back to regions_gdf
    regions_with_counts = regions_gdf.copy()
    regions_with_counts = regions_with_counts.merge(
        accident_counts, left_index=True, right_index=True, how="left"
    )

    # Fill regions with 0 accidents
    regions_with_counts["accident_count"] = regions_with_counts["accident_count"].fillna(0).astype(int)

    return regions_with_counts


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
        # sum(rate_r * pop_r) / sum(pop_r)
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


def main():
    # Base directory (project root, one level up from src)
    BASE_DIR = Path(__file__).resolve().parent.parent

    germany_geo_path = BASE_DIR / "data" / "processed" / "geo_data" / "Germany_merged.geojson"
    germany_acc_path = BASE_DIR / "data" / "processed" / "reduced_uk_dataset" / "modified_ger.csv"
    ger_regions_gdf = gpd.read_file(germany_geo_path)

    ger_accidents_gdf = load_accidents(
        str(germany_acc_path),
        category_filters={
            "casualty_severity": [1],
            "is_mcyle": [1]
        }
    )
    ger_regions_with_accidents = add_accident_counts_to_regions(
        regions_gdf=ger_regions_gdf,
        accidents_gdf=ger_accidents_gdf
    )

    # UK datasets
    uk_geo_path = BASE_DIR / "data" / "processed" / "geo_data" / "UK_merged.geojson"
    # uk_acc_path = BASE_DIR / "data" / "processed" / "reduced_uk_dataset" / "reduced_uk_dataset.csv"
    uk_acc_path = BASE_DIR / "data" / "processed" / "reduced_uk_dataset" / "reduced_uk_dataset.csv"
    os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"  # No size limit
    uk_regions_gdf = gpd.read_file(uk_geo_path)

    uk_accidents_gdf = load_accidents(
        str(uk_acc_path),
        category_filters={
            "collision_severity": [1],
            "is_mcyle": [1]
        }
    )
    print("UK accidents CRS:", uk_accidents_gdf.crs)
    print("UK regions CRS:", uk_regions_gdf.crs)

    uk_regions_with_accidents = add_accident_counts_to_regions(
        regions_gdf=uk_regions_gdf,
        accidents_gdf=uk_accidents_gdf
    )

    print(uk_regions_with_accidents['accident_count'].describe())
    print(uk_regions_with_accidents['accident_count'].value_counts().head())

    def print_population_stats(germany_gdf, uk_gdf):
        print("\n================ Germany Population Stats ================")
        print("Count:", len(germany_gdf))
        print("Min:", germany_gdf["population"].min())
        print("Median:", germany_gdf["population"].median())
        print("Mean:", germany_gdf["population"].mean())
        print("Max:", germany_gdf["population"].max())
        print(germany_gdf["population"].describe())

        print("\n================ UK Population Stats =====================")
        print("Count:", len(uk_gdf))
        print("Min:", uk_gdf["population"].min())
        print("Median:", uk_gdf["population"].median())
        print("Mean:", uk_gdf["population"].mean())
        print("Max:", uk_gdf["population"].max())
        print(uk_gdf["population"].describe())

        print("\nDone.\n")

    print_population_stats(ger_regions_with_accidents, uk_regions_with_accidents)

    bayes_df = build_bayes_dataset(ger_regions_with_accidents, uk_regions_with_accidents)

    mcmc = run_numpyro_model(bayes_df)
    df_probs = get_region_probs(mcmc, bayes_df)
    plot_expected_accidents(df_probs, mcmc)

    country_post = compute_country_posteriors(mcmc, bayes_df)
    plot_country_posteriors(country_post)

if __name__ == "__main__":
    main()
