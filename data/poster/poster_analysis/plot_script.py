"""Script to generate all the plots that are in the poster."""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "arial"

parent_data_path = pathlib.Path(__file__).parent.parent.resolve()


def get_cross_correlation(dataframe: pd.DataFrame):
    dataframe["linear_cross_correlation_mean"] = dataframe[
        "linear_cross_correlation_mean"
    ].apply(lambda x: list(map(float, x[1:-1].split())))
    dataframe["linear_cross_correlation_std"] = dataframe[
        "linear_cross_correlation_std"
    ].apply(lambda x: list(map(float, x[1:-1].split())))

    noise_intensity = dataframe["noise_intensity"].to_numpy()
    linear_cross_correlation_mean = np.zeros((len(dataframe.head(1)["linear_cross_correlation_mean"].values[0]), len(dataframe.index)))
    linear_cross_correlation_std = np.zeros((len(dataframe.head(1)["linear_cross_correlation_std"].values[0]), len(dataframe.index)))

    for i, dataframe_index in enumerate(dataframe.index):
        linear_cross_correlation_mean[:, i] = np.asarray(
            dataframe["linear_cross_correlation_mean"][dataframe_index]
        )
        linear_cross_correlation_std[:, i] = np.asarray(
            dataframe["linear_cross_correlation_std"][dataframe_index]
        )

    return noise_intensity, linear_cross_correlation_mean, linear_cross_correlation_std


def isotropic_white_vs_correlated_noise():
    white_metrics = pd.read_csv(
        parent_data_path / "white_isotropic_coherence_metrics.csv"
    )
    correlated_metrics = pd.read_csv(
        parent_data_path / "correlated_isotropic_coherence_metrics.csv"
    )

    min_diffusion_constant = [0.44, 0.59, 0.74, 0.89, 1.04, 1.19]
    max_diffusion_constant = [0.46, 0.61, 0.76, 0.91, 1.06, 1.21]
    diffusion_constant_array = [0.45, 0.60, 0.75, 0.90, 1.05, 1.20]

    colour_array = ["#b45114", "#15616d", "#006e4a", "#606e00", "#e6007e"]

    figure = plt.figure(layout="constrained", figsize=(6, 3))
    ax_top = figure.subplots(1, 2)

    temporal_correlation_value = 5.0
    min_temporal_correlation = 4.9
    max_temporal_correlation = 5.1

    spatial_correlation_value = 0.5
    min_spatial_correlation = 0.4
    max_spatial_correlation = 0.6

    for diffusion_index in range(0, len(diffusion_constant_array), 2):
        white_filtered = white_metrics[
            (
                white_metrics["diffusion_constant_xx"]
                > min_diffusion_constant[diffusion_index]
            )
            & (
                white_metrics["diffusion_constant_xx"]
                < max_diffusion_constant[diffusion_index]
            )
        ]
        (
            noise_intensity_white,
            linear_cross_correlation_mean_white,
            linear_cross_correlation_std_white,
        ) = get_cross_correlation(white_filtered)
        correlated_filtered = correlated_metrics[
            (
                correlated_metrics["diffusion_constant_xx"]
                > min_diffusion_constant[diffusion_index]
            )
            & (
                correlated_metrics["diffusion_constant_xx"]
                < max_diffusion_constant[diffusion_index]
            )
            & (correlated_metrics["temporal_correlation"] > min_temporal_correlation)
            & (correlated_metrics["temporal_correlation"] < max_temporal_correlation)
            & (correlated_metrics["spatial_correlation"] > min_spatial_correlation)
            & (correlated_metrics["spatial_correlation"] < max_spatial_correlation)
        ]
        (
            noise_intensity_correlated,
            linear_cross_correlation_mean_correlated,
            linear_cross_correlation_std_correlated,
        ) = get_cross_correlation(correlated_filtered)


        ax_top[0].errorbar(
            noise_intensity_white,
            linear_cross_correlation_mean_white[0, :],
            yerr=linear_cross_correlation_std_white[0, :],
            color=colour_array[diffusion_index],
            linestyle="",
            alpha=0.2,
            capsize=3,
        )
        ax_top[0].plot(
            noise_intensity_white,
            linear_cross_correlation_mean_white[0, :],
            color=colour_array[diffusion_index],
            marker=".",
            markersize=4,
        )
        ax_top[0].set_title("White noise", fontsize=15)
        ax_top[1].errorbar(
            noise_intensity_correlated,
            linear_cross_correlation_mean_correlated[0, :],
            yerr=linear_cross_correlation_std_correlated[0, :],
            color=colour_array[diffusion_index],
            linestyle="",
            alpha=0.2,
            capsize=3,
        )
        ax_top[1].plot(
            noise_intensity_correlated,
            linear_cross_correlation_mean_correlated[0, :],
            color=colour_array[diffusion_index],
            marker=".",
            markersize=4,
            label=f"D={diffusion_constant_array[diffusion_index]}",
        )
        ax_top[1].set_title(
            "$\\lambda$="
            + f"{spatial_correlation_value}"
            + ", $\\tau$="
            + f"{temporal_correlation_value}",
            fontsize=15,
        )
        ax_top[1].legend(bbox_to_anchor=(1.0, 1.0), loc="upper right", fontsize=8)

        for i in range(2):
            ax_top[i].set_xlabel("$\\sigma$", fontsize=15)
            ax_top[i].set_ylabel("S", fontweight="bold", fontsize=15)
            ax_top[i].tick_params(axis="both", labelsize=8)
            ax_top[i].grid(True, alpha=0.1)
    plt.show()


def isotropic_correlated_tau_lambda():
    correlated_metrics = pd.read_csv(
        parent_data_path / "correlated_isotropic_coherence_metrics.csv"
    )

    min_diffusion_constant = [0.44, 0.59, 0.74, 0.89, 1.04, 1.19]
    max_diffusion_constant = [0.46, 0.61, 0.76, 0.91, 1.06, 1.21]
    diffusion_constant_array = [0.45, 0.60, 0.75, 0.90, 1.05, 1.20]

    min_spatial_correlation = [0.49, 0.99, 2.99, 4.99, 9.99]
    max_spatial_correlation = [0.51, 1.01, 3.01, 5.01, 10.01]
    spatial_correlation_array = [0.5, 1.0, 3.0, 5.0, 10.0]

    min_temporal_correlation = [0.04, 0.09, 0.49, 0.99, 2.99, 4.99]
    max_temporal_correlation = [0.06, 0.11, 0.51, 1.01, 3.01, 5.01]
    temporal_correlation_array = [0.05, 0.1, 0.5, 1.0, 3.0, 5.0]

    colour_array = ["#7f055f", "#5957b9", "#0091de", "#00bfbf", "#5be081", "#f7ef66"]

    figure = plt.figure(layout="constrained", figsize=(8, 8))
    subfigs = figure.subfigures(2, 1, wspace=0.05)

    ax_top = subfigs[0].subplots(1, 2, sharey=True)
    for temporal_index in (1, 5):
        if temporal_index == 1:
            ax_index = 0
        else:
            ax_index = 1
        
        diffusion_index = 2
        for spatial_index in range(len(spatial_correlation_array)):
            filtered_coherence_metrics = correlated_metrics[
                (
                    correlated_metrics["diffusion_constant_xx"]
                    > min_diffusion_constant[diffusion_index]
                )
                & (
                    correlated_metrics["diffusion_constant_xx"]
                    < max_diffusion_constant[diffusion_index]
                )
                & (
                    correlated_metrics["temporal_correlation"]
                    > min_temporal_correlation[temporal_index]
                )
                & (
                    correlated_metrics["temporal_correlation"]
                    < max_temporal_correlation[temporal_index]
                )
                & (
                    correlated_metrics["spatial_correlation"]
                    > min_spatial_correlation[spatial_index]
                )
                & (
                    correlated_metrics["spatial_correlation"]
                    < max_spatial_correlation[spatial_index]
                )
            ]
            (noise_intensity, linear_cross_correlation_mean, linear_cross_correlation_std) = get_cross_correlation(filtered_coherence_metrics)


            ax_top[ax_index].errorbar(
                noise_intensity,
                linear_cross_correlation_mean[0, :],
                linear_cross_correlation_std[0, :],
                linestyle="",
                color=colour_array[spatial_index],
                alpha=0.2,
                capsize=3,
            )
            ax_top[ax_index].plot(
                noise_intensity,
                linear_cross_correlation_mean[0, :],
                label="$\\lambda$=" + f"{spatial_correlation_array[spatial_index]}",
                color=colour_array[spatial_index],
                marker=".",
                markersize=4,
            )
            ax_top[ax_index].set_title(
                "$\\tau$=" + f"{temporal_correlation_array[temporal_index]}",
                fontsize=15,
            )
            ax_top[ax_index].grid(True, alpha=0.1)
            ax_top[ax_index].set_xlabel("$\\sigma$", fontsize=15)
            ax_top[ax_index].tick_params(axis="both", labelsize=8)
    ax_top[1].legend(bbox_to_anchor=(1.0, 1.0), loc="upper left", fontsize=10)
    ax_top[0].set_ylabel("S", fontweight="bold", fontsize=15)

    ax_bottom = subfigs[1].subplots(1, 2)
    for spatial_index in (0, 4):
        if spatial_index == 0:
            ax_index = 0
        else:
            ax_index = 1
        diffusion_index = 2
        for temporal_index in range(len(temporal_correlation_array)):
            filtered_coherence_metrics = correlated_metrics[
                (
                    correlated_metrics["diffusion_constant_xx"]
                    > min_diffusion_constant[diffusion_index]
                )
                & (
                    correlated_metrics["diffusion_constant_xx"]
                    < max_diffusion_constant[diffusion_index]
                )
                & (
                    correlated_metrics["temporal_correlation"]
                    > min_temporal_correlation[temporal_index]
                )
                & (
                    correlated_metrics["temporal_correlation"]
                    < max_temporal_correlation[temporal_index]
                )
                & (
                    correlated_metrics["spatial_correlation"]
                    > min_spatial_correlation[spatial_index]
                )
                & (
                    correlated_metrics["spatial_correlation"]
                    < max_spatial_correlation[spatial_index]
                )
            ]

            noise_intensity, linear_cross_correlation_mean, linear_cross_correlation_std = get_cross_correlation(filtered_coherence_metrics)
            ax_bottom[ax_index].errorbar(
                noise_intensity,
                linear_cross_correlation_mean[0, :],
                yerr=linear_cross_correlation_std[0, :],
                linestyle="",
                color=colour_array[temporal_index],
                alpha=0.2,
                capsize=3,
            )
            ax_bottom[ax_index].plot(
                noise_intensity,
                linear_cross_correlation_mean[0, :],
                label="$\\tau$=" + f"{temporal_correlation_array[temporal_index]}",
                color=colour_array[temporal_index],
                marker=".",
                markersize=4,
            )
            ax_bottom[ax_index].set_title(
                "$\\lambda$=" + f"{spatial_correlation_array[spatial_index]}",
                fontsize=15,
            )
            ax_bottom[ax_index].grid(True, alpha=0.1)
            ax_bottom[ax_index].set_xlabel("$\\sigma$", fontsize=15)
            ax_bottom[ax_index].tick_params(axis="both", labelsize=8)
    ax_bottom[1].legend(bbox_to_anchor=(1.0, 1.0), loc="upper left", fontsize=10)
    ax_bottom[0].set_ylabel("S", fontweight="bold", fontsize=15)

    plt.show()


def correlated_anisotropic_plot():
    correlated_anisotropic_metrics = pd.read_csv(
        parent_data_path / "correlated_anisotropic_coherence_metrics.csv"
    )

    diffusion_xx_array = [1.50, 1.25, 1.0, 0.75]
    diffusion_yy_array = [0.0, 0.25, 0.5, 0.75]

    min_diffusion_xx = [1.40, 1.12, 0.9, 0.65]
    max_diffusion_xx = [1.60, 1.35, 1.1, 0.85]

    min_spatial_correlation = [0.49, 0.99, 2.99, 4.99, 9.99]
    max_spatial_correlation = [0.51, 1.01, 3.01, 5.01, 10.01]
    spatial_correlation_array = [0.5, 1.0, 3.0, 5.0, 10.0]

    min_temporal_correlation = [0.04, 0.09, 0.49, 0.99, 2.99, 4.99]
    max_temporal_correlation = [0.06, 0.11, 0.51, 1.01, 3.01, 5.01]
    temporal_correlation_array = [0.05, 0.1, 0.5, 1.0, 3.0, 5.0]

    colour_array = [
        "#1D42A6",
        "#606e00",
        "#527127",
        "#4b723f",
        "#4c7152",
        "#546e60",
        "#e6007e",
    ]

    figure = plt.figure(layout="constrained", figsize=(10, 6))
    subfigs = figure.subfigures(1, 2, hspace=0.07, width_ratios=[1, 2])

    ax_top = subfigs[0].subplots(1, 1)
    subfigs[0].suptitle("$\\lambda$=0.5, $\\tau$=0.05", fontsize=15)
    spatial_index = 0
    temporal_index = 0
    for diffusion_index in range(1, 2):
        correlated_filtered = correlated_anisotropic_metrics[
            (
                correlated_anisotropic_metrics["diffusion_constant_xx"]
                > min_diffusion_xx[diffusion_index]
            )
            & (
                correlated_anisotropic_metrics["diffusion_constant_xx"]
                < max_diffusion_xx[diffusion_index]
            )
            & (
                correlated_anisotropic_metrics["spatial_correlation"]
                > min_spatial_correlation[spatial_index]
            )
            & (
                correlated_anisotropic_metrics["spatial_correlation"]
                < max_spatial_correlation[spatial_index]
            )
            & (
                correlated_anisotropic_metrics["temporal_correlation"]
                > min_temporal_correlation[temporal_index]
            )
            & (
                correlated_anisotropic_metrics["temporal_correlation"]
                < max_temporal_correlation[temporal_index]
            )
        ]

        noise_intensity, linear_cross_correlation_mean, linear_cross_correlation_std = (
            get_cross_correlation(correlated_filtered)
        )

        for i in range(1, linear_cross_correlation_mean.shape[0]):
            ax_top.errorbar(
                noise_intensity,
                linear_cross_correlation_mean[i, :],
                yerr=linear_cross_correlation_std[i, :],
                linestyle="",
                color=colour_array[i - 1],
                capsize=3,
                alpha=0.2,
            )
            ax_top.plot(
                noise_intensity,
                linear_cross_correlation_mean[i, :],
                label=f"{(i-1) * 15}" "$\\degree$",
                color=colour_array[i - 1],
                marker=".",
                markersize=4,
            )

        ax_top.set_xlabel("$\\sigma$", fontsize=15)
        ax_top.set_ylabel("S", fontweight="bold", fontsize=15)
        ax_top.tick_params(axis="both", labelsize=8)
        ax_top.grid(True, alpha=0.1)
        ax_top.legend(bbox_to_anchor=(1.0, 1.0), loc="upper right", fontsize=8)

    subfigs_3d = subfigs[1].subfigures(1, 1)
    for diffusion_index in range(1, 2):
        ax_middle = subfigs_3d.add_subplot(projection="3d")
        yes_label = False
        for temporal_index in range(len(temporal_correlation_array)):
            for spatial_index in range(len(spatial_correlation_array)):
                correlated_filtered = correlated_anisotropic_metrics[
                    (
                        correlated_anisotropic_metrics["diffusion_constant_xx"]
                        > min_diffusion_xx[diffusion_index]
                    )
                    & (
                        correlated_anisotropic_metrics["diffusion_constant_xx"]
                        < max_diffusion_xx[diffusion_index]
                    )
                    & (
                        correlated_anisotropic_metrics["spatial_correlation"]
                        > min_spatial_correlation[spatial_index]
                    )
                    & (
                        correlated_anisotropic_metrics["spatial_correlation"]
                        < max_spatial_correlation[spatial_index]
                    )
                    & (
                        correlated_anisotropic_metrics["temporal_correlation"]
                        > min_temporal_correlation[temporal_index]
                    )
                    & (
                        correlated_anisotropic_metrics["temporal_correlation"]
                        < max_temporal_correlation[temporal_index]
                    )
                ]
                (
                    noise_intensity,
                    linear_cross_correlation_mean,
                    linear_cross_correlation_std,
                ) = get_cross_correlation(correlated_filtered)

                index_noise_opt_0 = np.argmax(linear_cross_correlation_mean[1, :])
                index_noise_opt_90 = np.argmax(linear_cross_correlation_mean[7, :])

                markersize = 50
                if index_noise_opt_0 == index_noise_opt_90:
                    ax_middle.scatter(
                        temporal_correlation_array[temporal_index],
                        spatial_correlation_array[spatial_index],
                        noise_intensity[index_noise_opt_0],
                        marker="1",
                        color="grey",
                        s=markersize,
                    )
                    ax_middle.scatter(
                        temporal_correlation_array[temporal_index],
                        spatial_correlation_array[spatial_index],
                        noise_intensity[index_noise_opt_90],
                        marker="2",
                        color="grey",
                        s=markersize,
                    )
                else:
                    if not yes_label:
                        ax_middle.scatter(
                            temporal_correlation_array[temporal_index],
                            spatial_correlation_array[spatial_index],
                            noise_intensity[index_noise_opt_0],
                            marker="1",
                            color=colour_array[0],
                            s=markersize,
                            label="$0\\degree$",
                        )
                        ax_middle.scatter(
                            temporal_correlation_array[temporal_index],
                            spatial_correlation_array[spatial_index],
                            noise_intensity[index_noise_opt_90],
                            marker="2",
                            color=colour_array[6],
                            s=markersize,
                            label="$90\\degree$",
                        )
                        yes_label = True
                    else:
                        ax_middle.scatter(
                            temporal_correlation_array[temporal_index],
                            spatial_correlation_array[spatial_index],
                            noise_intensity[index_noise_opt_0],
                            marker="1",
                            color=colour_array[0],
                            s=markersize,
                        )
                        ax_middle.scatter(
                            temporal_correlation_array[temporal_index],
                            spatial_correlation_array[spatial_index],
                            noise_intensity[index_noise_opt_90],
                            marker="2",
                            color=colour_array[6],
                            s=markersize,
                        )
        ax_middle.set_xlabel("$\\tau$", fontsize=15)
        ax_middle.set_ylabel("$\\lambda$", fontsize=15)
        ax_middle.set_zlabel("$\\sigma_{opt}$", fontsize=15)

        ax_middle.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_middle.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_middle.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_middle.legend(fontsize=10)

    plt.show()



if __name__ == "__main__":
    isotropic_white_vs_correlated_noise()
    isotropic_correlated_tau_lambda()
    correlated_anisotropic_plot()
