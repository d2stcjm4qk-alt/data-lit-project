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
from matplotlib import rcParams


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
    accidents = df["accident_count"].to_numpy()

    # Exposure in a billion vehicle-km (monthly exposure)
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


def analyze_and_plot_country_seasonality(
        posterior,
        country_names=("Germany", "UK"),
        de_idx=0,
        uk_idx=1,
):
    """
    Analyze posterior seasonality strength and plot monthly effects by country.
    """

    # -----------------------------
    # LaTeX-style plotting settings
    # -----------------------------
    rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 28,
        "axes.titlesize": 30,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 24,
    })

    # Colors (intuitive, not oversaturated)
    color_de = "darkorange"  # soft orange for Germany
    color_uk = "royalblue"  # blue for UK

    # -----------------------------
    # Extract posterior quantities
    # -----------------------------
    month_eff = posterior["month_effect_country"]
    # shape: (samples, country, month)

    sigma_season = month_eff.std(axis=-1)  # (samples, country)

    sigma_de = sigma_season[:, de_idx]
    sigma_uk = sigma_season[:, uk_idx]

    # -----------------------------
    # Numerical summaries
    # -----------------------------
    print(f"{country_names[0]} seasonality (std over months):")
    print(np.percentile(sigma_de, [5, 50, 95]))

    print(f"\n{country_names[1]} seasonality (std over months):")
    print(np.percentile(sigma_uk, [5, 50, 95]))

    prob_de_more = np.mean(sigma_de > sigma_uk)
    print(
        f"\nP({country_names[0]} more seasonal than "
        f"{country_names[1]}) = {prob_de_more:.3f}"
    )

    # -----------------------------
    # Monthly posterior summaries
    # -----------------------------
    months = np.arange(12)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    mean_de = month_eff[:, de_idx, :].mean(axis=0)
    ci_de = np.percentile(month_eff[:, de_idx, :], [5, 95], axis=0)

    mean_uk = month_eff[:, uk_idx, :].mean(axis=0)
    ci_uk = np.percentile(month_eff[:, uk_idx, :], [5, 95], axis=0)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(9, 4.5))

    plt.plot(
        months,
        mean_de,
        label=country_names[0],
        color=color_de,
        linewidth=2.5,
    )
    plt.fill_between(
        months,
        ci_de[0],
        ci_de[1],
        color=color_de,
        alpha=0.25,
    )

    plt.plot(
        months,
        mean_uk,
        label=country_names[1],
        color=color_uk,
        linewidth=2.5,
    )
    plt.fill_between(
        months,
        ci_uk[0],
        ci_uk[1],
        color=color_uk,
        alpha=0.25,
    )

    plt.xticks(months, month_labels)
    plt.xlabel(r"Month")
    plt.ylabel(r"Country-level seasonal effect (log-rate)")
    plt.title(r"Posterior monthly seasonality by country")

    plt.legend(frameon=False)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def main():
    # Base directory (project root, one level up from src)
    BASE_DIR = Path(__file__).resolve().parent.parent

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

    # UK datasets
    uk_geo_path = BASE_DIR / "data" / "preprocessed" / "uk" / "traffic" / "uk_gdf_with_osm_roads.gpkg"
    uk_acc_path = BASE_DIR / "data" / "processed" / "reduced_uk_dataset" / "reduced_uk_dataset.csv"
    os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"  # No size limit
    uk_regions_gdf = gpd.read_file(uk_geo_path)

    uk_accidents_gdf = load_accidents(
        str(uk_acc_path),
        category_filters={
            "collision_severity": [1],
        }
    )

    uk_regions_with_accidents = add_accident_counts_to_regions_monthly(
        regions_gdf=uk_regions_gdf,
        accidents_gdf=uk_accidents_gdf
    )

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

    analyze_and_plot_country_seasonality(posterior)

    # TODO adjust seasons to metreological
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
