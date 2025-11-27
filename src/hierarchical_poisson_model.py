import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import contextily as ctx
from typing import Optional, Dict, Tuple
import pyproj
from shapely.ops import unary_union
from libpysal import weights

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import arviz as az

def build_bayes_dataset(germany_gdf, uk_gdf):
    """
    Extracts region_id, region_name, accidents, population
    and attaches a 'country' column so both datasets share the same structure.
    Returns a combined DataFrame ready for the Bayesian model.
    """
    print(f'GErmany before: {germany_gdf.head()}')
    # ==== Germany ====
    df_de = pd.DataFrame({
        "region_id": germany_gdf["region_code"].astype(str),
        #"region_name": germany_gdf["GEN"].astype(str),
        "accidents": germany_gdf["accident_count"].astype(int),
        "population": germany_gdf["population"].fillna(0).astype(int),
        "country": "Germany"
    })

    # ==== UK ====
    # LAD has LAD24CD (code) and LAD24NM (name)
    name_col = "LAD24NM" if "LAD24NM" in uk_gdf.columns else "GEN"

    df_uk = pd.DataFrame({
        "region_id": uk_gdf["region_code"].astype(str),
        #"region_name": uk_gdf[name_col].astype(str),
        "accidents": uk_gdf["accident_count"].astype(int),
        "population": uk_gdf["population"].fillna(0).astype(int),
        "country": "UK"
    })

    df = pd.concat([df_de, df_uk], ignore_index=True)
    print(f"Merged DE+UK regions for Bayesian model: {len(df)} rows")
    return df


# --- Poisson-Gamma hierarchical model ---
def hierarchical_poisson_model(accidents, population, country_idx, n_countries=2):
    n_regions = len(accidents)

    # Country-level hyperpriors
    mu_country = numpyro.sample("mu_country", dist.Normal(0, 5).expand([n_countries]))
    sigma_country = numpyro.sample("sigma_country", dist.HalfNormal(2).expand([n_countries]))

    # Region-level effects
    region_effect = numpyro.sample("region_effect", dist.Normal(0, 1).expand([n_regions]))

    # log lambda per person
    log_lambda = mu_country[country_idx] + sigma_country[country_idx] * region_effect

    # Poisson likelihood with log-population offset
    numpyro.sample(
        "obs",
        dist.Poisson(jnp.exp(log_lambda + jnp.log(population))),
        obs=accidents
    )


# ==========================
# Run NumPyro MCMC
# ==========================
def run_numpyro_scaled_model(df, num_warmup=1000, num_samples=2000):
    # Compute accident rates per 100k population
    accident_rate = (df["accidents"] / df["population"]) * 1e5
    accidents = accident_rate.fillna(0).to_numpy()
    population = df["population"].to_numpy()

    # Map countries
    df['country_idx'] = df['country'].map({'Germany': 0, 'UK': 1}).astype(int)
    country_idx = df['country_idx'].to_numpy()

    kernel = NUTS(hierarchical_poisson_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1)
    mcmc.run(jax.random.PRNGKey(0),
             accidents=accidents,
             population=population,
             country_idx=country_idx)
    mcmc.print_summary()
    return mcmc


def extract_rates_per_100k(mcmc, df):
    import arviz as az
    import numpy as np

    samples = mcmc.get_samples()
    mu_country = samples['mu_country']
    sigma_country = samples['sigma_country']
    region_effect = samples['region_effect']

    country_idx = df['country'].map({'Germany': 0, 'UK': 1}).to_numpy()
    n_samples, n_regions = region_effect.shape

    # Reconstruct log per-person rates
    log_lambda_samples = mu_country[:, country_idx] + sigma_country[:, country_idx] * region_effect.T
    lambda_samples = np.exp(log_lambda_samples)  # per-person rate

    # Convert to per 100k population
    lambda_100k = lambda_samples * 1e5

    # Posterior mean and 95% HDI
    rate_mean = lambda_100k.mean(axis=0)
    rate_hdi = az.hdi(lambda_100k, hdi_prob=0.95)

    df_out = df.copy()
    df_out['rate_mean'] = rate_mean
    df_out['rate_lower'] = rate_hdi[:, 0]
    df_out['rate_upper'] = rate_hdi[:, 1]

    return df_out


# ==========================
# Extract posterior expected rates
# ==========================
def extract_rates(mcmc, df):
    samples = mcmc.get_samples()
    mu_country = samples['mu_country']  # (num_samples, n_countries)
    sigma_country = samples['sigma_country']
    region_effect = samples['region_effect']  # (num_samples, n_regions)

    country_idx = df['country'].map({'Germany': 0, 'UK': 1}).to_numpy()

    # log rate per person
    log_lambda_samples = mu_country[:, country_idx] + sigma_country[:, country_idx] * region_effect.T
    lambda_samples = np.exp(log_lambda_samples) * 1e5  # per 100k population

    import arviz as az
    rate_mean = lambda_samples.mean(axis=0)
    rate_hdi = az.hdi(lambda_samples, hdi_prob=0.95)

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
    lambda_100k = lambda_samples * 1e5

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
        print(f'###################{len(df_probs)}')
        x_min, x_max = 0, len(df_probs) / 2

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


def main():
    germany_acc_path = r"C:\Users\pbaue\Documents\Master_Informatik\Data_Literacy\Unfallatlas\csv\Unfallorte2024_LinRef.csv"
    germany_geo_path = r"C:\Users\pbaue\Documents\Master_Informatik\Data_Literacy\GeoData\Germany_merged.geojson"
    kreise = gpd.read_file(germany_geo_path)

    accidents = load_accidents(
        germany_acc_path,
        category_filters={
            "UKATEGORIE": [1],  # deadly
            "IstKrad": [1]  # only darkness
        }
    )