"""File for prototype functions to analyse the structure functions."""
import numpy as np
import pathlib
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from src.helper.datatypes import SimulationParameters, DiffusionType
from src.prototype.prototype_spatiotemporal_metrics import (
    calculate_circular_line_integral,
    calculate_elliptic_line_integral,
    calculate_ellipse_perimeter,
)
from src.helper.plot_3d import plot_3d_surface
from src.prototype.prototype_fit_ellipse import fit_ellipse_to_power_spectra
from src.helper.helper_save import configure_saving_strings



def analyse_spatial_wave_number(
    ensemble_mean_power_spectra: np.ndarray,
    simulation_parameters: SimulationParameters,
    path_to_save: pathlib.Path,
    angular_step_size: float = np.pi / 64,
) -> np.ndarray:
    """
    Calculate the spatial wave number by doing a circular or elliptical line integral over the
    average structure function.

    Args:
        ensemble_mean_power_spectra (np.ndarray): the mean power spectra for this condition
        simulation_parameters (SimulationParameters): simulation parameters
        path_to_save (pathlib.Path): path to save the elliptical parameters, if anisotropic
        angular_step_size (float, optional): angular step size for integration. Defaults to
            np.pi/64.

    Returns:
        np.ndarray: the average structure function
    """
    
    average_structure_function_array = np.empty(
        int(simulation_parameters.discretisation_parameters.grid_size[0] / 2) - 1
    )
    for k in range(
        int(simulation_parameters.discretisation_parameters.grid_size[0] / 2) - 1
    ):
        # Use k+1 because having either a circle or an ellipse with its radii 0 is nothing
        if simulation_parameters.diffusion_type == DiffusionType.ISOTROPIC:
            # For isotropic, force to be circular
            average_structure_function_array[k] = calculate_circular_line_integral(
                ensemble_mean_power_spectra,
                simulation_parameters.discretisation_parameters.grid_size,
                simulation_parameters.discretisation_parameters.spatial_step_size,
                k + 1,
                angular_step_size,
            ) / (2 * np.pi * (k + 1))
        elif simulation_parameters.diffusion_type == DiffusionType.ANISOTROPIC:
            # For anisotropic, first fit the ellipse to the power spectra
            ellipse_parameter = fit_ellipse_to_power_spectra(
                ensemble_mean_power_spectra
            )

            # Save the ellipse parameter into json in the folder
            diffusion_string, noise_intensity_string = configure_saving_strings(
                simulation_parameters
            )
            path_ellipse = (
                path_to_save
                / f"ellipse_param_{diffusion_string}_{noise_intensity_string}.json"
            )
            ellipse_parameter.to_json(path_ellipse)

            # Then calculate the average structure function by doing a path integral over the
            # (scaled) ellipse
            average_structure_function_array[k] = calculate_elliptic_line_integral(
                ensemble_mean_power_spectra,
                simulation_parameters.discretisation_parameters.grid_size,
                simulation_parameters.discretisation_parameters.spatial_step_size,
                ellipse_parameter,
                k + 1,
                angular_step_size,
            ) / calculate_ellipse_perimeter(ellipse_parameter, scale_factor=k + 1)

    return average_structure_function_array


def find_peaks_in_almost_monotonically_decreasing(
    x_values: np.ndarray,
    y_values: np.ndarray,
    prominence: float = 0.1,
    start_index_peaks: int = 1,
) -> np.ndarray:
    """
    This is a function to find peaks in a line that is almost monotonically decreasing. Because the
    peaks are not very high (or sometimes not even a proper local maxima), a different approach
    is required here by taking the derivative of the line and then finding the peaks of
    the derivative.
    Args:
        x_values (np.ndarray): 1D array of the x values of the function
        y_values (np.ndarray): 1D array of the y values of the function
        prominence (float, optional): Prominence for finding the peaks of the derivative.
            Defaults to 0.1.
        start_index_peaks (int, optional): The starting index for finding peaks, as the first
            derivative might not be representative. Defaults to 1.

    Returns:
        np.ndarray : an array of the index of the peaks in the given y_values array
    """
    # Check that the x_values and y_values have the same shape
    assert len(x_values) == len(y_values)

    # Find delta_x, and then check that they are all approximately the same
    delta_x_array = x_values[1:] - x_values[:-1]
    assert np.all(
        np.logical_and(
            delta_x_array < 1.1 * delta_x_array[0],
            delta_x_array > 0.9 * delta_x_array[0],
        )
    )
    # Use central difference to find the derivative of y
    y_prime = (y_values[2:] - y_values[:-2]) / delta_x_array[0]

    # Find the peaks in y derivative
    peaks_in_y_prime = find_peaks(
        y_prime, prominence=prominence * np.max(np.abs(y_prime[start_index_peaks:]))
    )[0]

    # Add 2 to the index because of the calculation for the derivative of y-values
    return peaks_in_y_prime + 2


def save_power_spectra(
    ensemble_mean_power_spectra: np.ndarray,
    simulation_parameters: SimulationParameters,
    wave_number: np.ndarray,
    path_to_save: pathlib.Path,
    title: str,
):
    # Save image
    diffusion_string, noise_intensity_string = configure_saving_strings(
        simulation_parameters
    )

    fig, ax = plt.subplots()
    ax.imshow(
        ensemble_mean_power_spectra,
        cmap="jet",
        norm="log",
        extent=(wave_number[0], wave_number[-1], wave_number[0], wave_number[-1]),
    )
    ax.set_title(f"Power spectra for {title}")
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")

    plt.savefig(
        path_to_save
        / f"power_spectra_im_{diffusion_string}_{noise_intensity_string}.png"
    )
    plt.clf()

    three_d_fig = plot_3d_surface(
        wave_number,
        wave_number,
        np.log10(ensemble_mean_power_spectra),
        "kx",
        "ky",
        "log10(power spectra)",
    )
    plt.title(f"Power spectra for {title}")
    plt.savefig(path_to_save / f"power_spectra_surf_{diffusion_string}_{noise_intensity_string}.png")
    plt.close()
