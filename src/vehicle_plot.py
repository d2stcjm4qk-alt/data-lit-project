import os
import calendar
from pathlib import Path
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles
# Import analysis modules
from hierarchical_seasonal_analysis import (
    AccidentDataProcessor,
    build_bayes_dataset,
    run_mcmc_analysis,
)


# plotting
def plot_seasonal_index(
        posterior_mb,
        posterior_c,
        country_names=("Germany", "UK"),
        mode_names=("Motorcycles", "Cars"),
):
    # Prepare datasets
    datasets = [posterior_mb["month_effect_country"], posterior_c["month_effect_country"]]

    # tueplot settings
    plt.rcParams.update(bundles.icml2024(column="half", nrows=1, ncols=2, usetex=True))

    # Create subplots
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    colors = ["darkorange", "royalblue"]
    month_nums = np.arange(12)
    month_labels = [calendar.month_abbr[i + 1] for i in month_nums]
    lines = []
    
    # Plot for each vehicle mode
    for i, mode_name in enumerate(mode_names):
        ax = axes[i]
        current_data = datasets[i]

        for j, country in enumerate(country_names):
            # Compute relative risk (exponentiate log-scale posterior mean)
            mean_log = current_data[:, j, :].mean(axis=0)
            index = np.exp(mean_log)  
            
            line, = ax.plot(month_nums, index, color=colors[j], linewidth=2.0, label=country)
            
            # Store legend handles from the first subplot only
            if i == 0: lines.append(line)  

            ax.plot(
                month_nums,
                index,
                color=colors[j],
                linewidth=1.0,
                label=country,
            )

            # Highlight deviation from baseline
            ax.fill_between(
                month_nums,
                index,
                1.0,
                color=colors[j],
                alpha=0.1,
                edgecolor="none"
            )

            # Highlight the peak risk month
            # p_idx = index.argmax()
            # p_val = index.max()
            # ax.scatter(p_idx, p_val, color=colors[j], s=10, zorder=5)
            
            # Label annotation positioning
            # if i == 0:
            #     ax.annotate(
            #         f"{p_val:.2f}",
            #         (p_idx, p_val),
            #         xytext=(-1, -8),
            #         textcoords="offset points",
            #         ha="center",
            #         fontsize=6,
            #         color=colors[j]
            #     )
            # else:
            #     ax.annotate(
            #         f"{p_val:.2f}",
            #         (p_idx, p_val),
            #         xytext=(0, 3),
            #         textcoords="offset points",
            #         ha="center",
            #         fontsize=6,
            #         color=colors[j])

        # Add baseline reference (Relative Risk = 1.0)
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=1)
        ax.set_title(f"{mode_name}", loc="center")
        ax.set_xticks(month_nums[::2])
        ax.set_xticklabels(month_labels[::2])
        ax.grid(True, alpha=0.15, ls=":")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if i == 0:
            ax.set_ylabel(r"Relative Risk" + "\n" + r"$\exp(\gamma_{c[j],m})$")

        # Shared legend configuration
        fig.legend(
            handles=lines,
            labels=list(country_names),
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.15)
        )

    # Export figure
    plt.tight_layout()
    BASE_DIR = Path(__file__).resolve().parent.parent
    plot_path= BASE_DIR / 'results'/"figures" / 'vehicle_seasonality.pdf'
    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()



# main
# here we use the same functions as in hierarchical_seasonal_analysis.py
def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

    proc = AccidentDataProcessor()

    # Motorbikes
    ger_acc_mb = proc.load_accidents(
        BASE_DIR / "data" / "preprocessed" / "germany" / "collisions"/ "preprocessed_ger.csv",
        category_filters={"casualty_severity": [1], "is_motorcycle": [1]},
    )

    uk_acc_mb = proc.load_accidents(
        BASE_DIR / "data" / "preprocessed" / "uk" / "collisions"/ "preprocessed_uk.csv",
        category_filters={"casualty_severity": [1], "is_motorcycle": [1]},
    )

    ger_regions = gpd.read_file(
        BASE_DIR / "data/preprocessed/germany/geofiles/ger_gdf_with_osm_roads.gpkg"
    )
    uk_regions = gpd.read_file(
        BASE_DIR / "data/preprocessed/uk/geofiles/uk_gdf_with_osm_roads.gpkg"
    )

    ger_mb = proc.aggregate_by_region_monthly(ger_regions, ger_acc_mb)
    uk_mb = proc.aggregate_by_region_monthly(uk_regions, uk_acc_mb)

    bayes_mb = build_bayes_dataset(ger_mb, uk_mb)
    posterior_mb = run_mcmc_analysis(bayes_mb)


    # Cars
    ger_acc_c = proc.load_accidents(
        BASE_DIR / "data" / "preprocessed" / "germany" / "collisions"/ "preprocessed_ger.csv",
        category_filters={"casualty_severity": [1], "is_car": [1]},
    )

    uk_acc_c = proc.load_accidents(
        BASE_DIR / "data" / "preprocessed" / "uk" / "collisions"/ "preprocessed_uk.csv",
        category_filters={"casualty_severity": [1], "is_car": [1]},
    )

    ger_c = proc.aggregate_by_region_monthly(ger_regions, ger_acc_c)
    uk_c = proc.aggregate_by_region_monthly(uk_regions, uk_acc_c)

    bayes_c = build_bayes_dataset(ger_c, uk_c)
    posterior_c = run_mcmc_analysis(bayes_c)
    

    # samples_path_mb = BASE_DIR / "data/mcmc/mcmc_samples_region_motorbike.npz"
    # data_mb = np.load(samples_path_mb)
    # posterior_mb = {k: data_mb[k] for k in data_mb.files}

    # samples_path_c = BASE_DIR / "data/mcmc/mcmc_samples_region_car.npz"
    # data_c = np.load(samples_path_c)
    # posterior_c = {k: data_c[k] for k in data_c.files}

    # plotting
    plot_seasonal_index(posterior_mb, posterior_c)


if __name__ == "__main__":
    main()