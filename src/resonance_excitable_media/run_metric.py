"""For calculating the metrics for multiple conditions (batch)."""
from multiprocessing import Pool, cpu_count
from itertools import repeat, product

import numpy as np
import pathlib
import pandas as pd

from src.helper.datatypes import (
    FitzHughNagumoConstants,
    DiscretisationParameters,
    SimulationParameters,
    NoiseType,
    DiffusionType,
    DiffusionTensor,
    ParametersBatch,
    SimulationType,
)
from src.resonance_excitable_media.run_single import run_single
from src.helper.animate import animate_plot, save_animation
from src.resonance_excitable_media.create_noise import (
    generate_white_noise,
    generate_spatiotemporal_correlated_noise,
)

from src.helper.spatiotemporal_metrics import (
    calculate_structure_function,
    calculate_circular_line_integral,
    calculate_elliptic_line_integral,
    calculate_ellipse_perimeter,
    calculate_linear_cross_correlation,
)

from src.helper.fit_ellipse import fit_ellipse_to_power_spectra

####################################################################################################
### Helper functions
####################################################################################################


def configure_saving_strings(
    simulation_parameters: SimulationParameters,
) -> tuple[str, str]:
    """Convenience function to generate strings for saving purposes.

    Args:
        simulation_parameters (SimulationParameters): simulation parameters

    Returns:
        diffusion_string (str): string of the diffusion
        noise_intensity_string (str): string of the noise intensity
    """
    # Configure out some strings to make descriptive file names
    if simulation_parameters.diffusion_type == DiffusionType.ISOTROPIC:
        diffusion_string = f"isotropic_diffusion_constant_{simulation_parameters.diffusion_constant:.2f}"
    elif simulation_parameters.diffusion_type == DiffusionType.ANISOTROPIC:
        assert isinstance(simulation_parameters.diffusion_constant, DiffusionTensor)
        diffusion_string = ""
        diffusion_string = diffusion_string.join(
            (
                f"difussion_xx_{simulation_parameters.diffusion_constant.d_xx[0,0]}",
                f"_xy_{simulation_parameters.diffusion_constant.d_xy[0,0]}",
                f"_yy{simulation_parameters.diffusion_constant.d_yy[0, 0]}",
            )
        )
    else:
        diffusion_string = (
            f"diffusion_{simulation_parameters.diffusion_type.name.lower()}"
        )
    noise_intensity_string = f"noise_{simulation_parameters.noise_intensity}"

    return diffusion_string, noise_intensity_string


def save_array(
    path_to_save: pathlib.Path,
    simulation_parameters: SimulationParameters,
    ensemble_mean_power_spectra: np.ndarray,
    average_structure_function_array: np.ndarray,
) -> None:
    """Save the calculated power spectra and structure function to csv file.

    Args:
        path_to_save (pathlib.Path): path to the folder to save the arrays
        simulation_parameters (SimulationParameters): simulation parameters
        ensemble_mean_power_spectra (np.ndarray): calculated (mean) power spectra
        average_structure_function_array (np.ndarray): calculated structure function
    """
    diffusion_string, noise_intensity_string = configure_saving_strings(
        simulation_parameters
    )
    power_spectra_path = (
        path_to_save / f"power_spectra_{diffusion_string}_{noise_intensity_string}.csv"
    )
    structure_function_path = (
        path_to_save / f"p_k_array_{diffusion_string}_{noise_intensity_string}.csv"
    )

    np.savetxt(power_spectra_path, ensemble_mean_power_spectra, delimiter=",")
    np.savetxt(structure_function_path, average_structure_function_array, delimiter=",")


def save_coherence_metrics(path_to_save: pathlib.Path, dataframe: pd.DataFrame) -> None:
    """Save the coherence metrics to file. If the file already exists, then append.

    Args:
        path_to_save (pathlib.Path): path to the file to save the coherence metrics.
        dataframe (pd.DataFrame): the coherence metrics as a pandas dataframe
    """
    if path_to_save.exists():
        dataframe.to_csv(path_to_save, mode="a", index=False, header=False)
    else:
        dataframe.to_csv(path_to_save, mode="w", index=False, header=True)


####################################################################################################
### Main running functions
####################################################################################################


def run_ensemble_and_analyse(
    ensemble_number: int,
    simulation_parameters: SimulationParameters,
    simulation_type: SimulationType,
    animation_save: bool,
    path_to_save: pathlib.Path,
) -> tuple[np.ndarray, float, float]:
    """
    Run an ensemble run (i.e. multiple times for a single conditions) and then calculate the
    average metrics.

    Args:
        ensemble_number (int): the number of runs for this ensemble
        simulation_parameters (SimulationParameters): simulation parameters to run the simulation
        simulation_type (SimulationType): the type of simulation. can be noise only,
            can be the full run
        animation_save (bool): whether or not to save the animation
        path_to_save (pathlib.Path): path to save the results in

    Returns:
        tuple[np.ndarray, float, float]: _description_
    """
    # Get a random integer within (0, ensemble number-10) to save as a representative animation of
    # this condition. Use ensemble number-10 just in case there's an infinite case and we can move
    # to the next integer
    animation_index = np.random.randint(low=0, high=ensemble_number - 10)

    # Reserve arrays for the result.
    ensemble_mean_power_spectra_array = np.empty(
        (
            simulation_parameters.discretisation_parameters.grid_size[0],
            simulation_parameters.discretisation_parameters.grid_size[1],
            ensemble_number,
        )
    )
    ensemble_linear_cross_correlation_array = np.empty(ensemble_number)

    # Keep track of the infinite failures. If too many failures, then just stop for this condition
    failure_count = 0
    for ensemble_index in range(ensemble_number):
        try:
            # If it's the full simulation, just call the run_single function
            if simulation_type == SimulationType.FULL_SIMULATION:
                result = run_single(simulation_parameters)
                membrane_potential = result.membrane_potential

            # If we're running for the noise only, then have to configure it
            else:
                number_of_time_steps = int(
                    simulation_parameters.simulation_time
                    / simulation_parameters.discretisation_parameters.temporal_step_size
                )
                if simulation_parameters.noise_type == NoiseType.WHITE:
                    membrane_potential = generate_white_noise(
                        noise_intensity=simulation_parameters.noise_intensity,
                        grid_size=simulation_parameters.discretisation_parameters.grid_size[
                            0
                        ],
                        number_of_time_steps=number_of_time_steps,
                    )
                elif simulation_parameters.noise_type == NoiseType.CORRELATED:
                    membrane_potential = generate_spatiotemporal_correlated_noise(
                        simulation_parameters.spatial_noise_correlation,
                        simulation_parameters.temporal_noise_correlation,
                        simulation_parameters.noise_intensity,
                        simulation_parameters.discretisation_parameters.spatial_step_size[
                            0
                        ],
                        simulation_parameters.discretisation_parameters.temporal_step_size,
                        simulation_parameters.discretisation_parameters.grid_size[0],
                        number_of_time_steps,
                    )
                else:
                    # Should never get here, but included to keep typechecker happy
                    membrane_potential = np.zeros(
                        (
                            simulation_parameters.discretisation_parameters.grid_size[
                                0
                            ],
                            simulation_parameters.discretisation_parameters.grid_size[
                                0
                            ],
                            number_of_time_steps,
                        )
                    )

            # If we're saving a representative animation, check if the index is the same as the
            # current index
            if animation_save and (ensemble_index == animation_index):
                # Animate and save the plot
                anim = animate_plot(membrane_potential, membrane_potential.shape[2])

                diffusion_string, noise_intensity_string = configure_saving_strings(
                    simulation_parameters
                )

                # Save the animation to the given folder and file name
                save_animation(
                    anim,
                    path_to_save
                    / f"anim_{diffusion_string}_{noise_intensity_string}.gif",
                )

            # Calculate the structure function (as given in Perc 2005). Use only the frames between
            # -20 and -10 index (10 from the 20 last frames) because the paper didn't really give
            # any given time, and this should take the 'stable' state of the simulation
            power_spectra = calculate_structure_function(
                membrane_potential,
                simulation_parameters.discretisation_parameters,
                -20,
                -10,
            )
            ensemble_mean_power_spectra_array[:, :, ensemble_index] = power_spectra

            # Calculate the linear cross correlation (Busch and Kaiser, 2003)
            ensemble_linear_cross_correlation_array[ensemble_index] = (
                calculate_linear_cross_correlation(
                    membrane_potential,
                    simulation_parameters.discretisation_parameters,
                )
            )
        except ValueError:
            # If there's a ValueError, that means that there's an infinite value in the temporal
            # integration. We kept track of the count, so if there's less than 20% failure, we
            # continue, but if there's more than 20% failure, we assume this condition will give
            # always infinite value, so just abort
            if failure_count < 0.2 * ensemble_number:
                ensemble_mean_power_spectra_array[:, :, ensemble_index] = (
                    np.empty(simulation_parameters.discretisation_parameters.grid_size)
                    * np.nan
                )
                ensemble_linear_cross_correlation_array[ensemble_index] = np.nan
                failure_count += 1

                # If we're saving the animation and it happens to be the one which is meant to be
                # saved, add one to the animation index
                if animation_save and (animation_index == ensemble_index):
                    animation_index += 1
                continue
            else:
                print("Too many infinite values, aborting for this condition!")
                break

    # Get the mean of the metrics, disregarding the NaN values.
    ensemble_mean_power_spectra = np.nanmean(ensemble_mean_power_spectra_array, axis=2)
    linear_cross_correlation_mean = np.nanmean(ensemble_linear_cross_correlation_array)
    linear_cross_correlation_std = np.nanstd(ensemble_linear_cross_correlation_array)
    return (
        ensemble_mean_power_spectra,
        float(linear_cross_correlation_mean),
        float(linear_cross_correlation_std),
    )


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


def run_ensemble_in_parallel(
    diffusion_type: DiffusionType,
    noise_type: NoiseType,
    fitzhugh_nagumo_constants: FitzHughNagumoConstants,
    discretisation_parameters: DiscretisationParameters,
    diffusion_constant_xx: float,
    diffusion_constant_xy: float,
    diffusion_constant_yy: float,
    simulation_time: float,
    noise_intensity: float,
    spatial_correlation: float,
    temporal_correlation: float,
    ensemble_number: int,
    path_to_save: pathlib.Path,
    animation_save: bool,
) -> np.ndarray:
    """
    Set up and do an ensemble run for the given condition. Structured so that this can be done via
    multiprocessing.

    Args:
        diffusion_type (DiffusionType): the diffusion type for this condition
        noise_type (NoiseType): the noise type for this condition
        fitzhugh_nagumo_constants (FitzHughNagumoConstants): FHN constants for this condition
        discretisation_parameters (DiscretisationParameters): discretisation parameters for this
            condition
        diffusion_constant_xx (float): For isotropic, this is the diffusion constant value. For
            anisotropic, this is the value for the xx diffusion tensor
        diffusion_constant_xy (float): For isotropic, put 0. For anisotropic, this is the value for
            the xy diffusion tensor
        diffusion_constant_yy (float): For isotropic, put 0. For anisotropic, this is the value for
            the yy diffusion tensor
        simulation_time (float): simulation time for this condition
        noise_intensity (float): noise intensity for this condition
        spatial_correlation (float): For white noise, put 0. For correlated noise, the value for
            the spatial correlation of the noise
        temporal_correlation (float): For white noise, put 0. For correlated noise, the value for
            the temporal correlation of the noise
        ensemble_number (int): the number of runs considered an ensemble
        path_to_save (pathlib.Path): path to the main folder to save the data
        animation_save (bool): whether or not to save a representative animation of this condition

    Returns:
        np.ndarray: an array of the coherence metrics
    """

    # If the diffusion type is isotropic, then just use the xx as the diffusion constant
    if diffusion_type == DiffusionType.ISOTROPIC:
        diffusion_constant = diffusion_constant_xx
    # If anisotropic, then first make the diffusiont tensor by the given constants
    elif diffusion_type == DiffusionType.ANISOTROPIC:
        d_xx_tensor = (
            np.ones((discretisation_parameters.grid_size)) * diffusion_constant_xx
        )
        d_xy_tensor = (
            np.ones((discretisation_parameters.grid_size)) * diffusion_constant_xy
        )
        d_yy_tensor = (
            np.ones((discretisation_parameters.grid_size)) * diffusion_constant_yy
        )

        diffusion_constant = DiffusionTensor(d_xx_tensor, d_xy_tensor, d_yy_tensor)
    else:
        # Should not get here, but to keep typecheck happy and as a last resort put in 0
        diffusion_constant = 0.0

    simulation_parameters = SimulationParameters(
        fitzhugh_nagumo_constants=fitzhugh_nagumo_constants,
        discretisation_parameters=discretisation_parameters,
        diffusion_type=diffusion_type,
        diffusion_constant=diffusion_constant,
        simulation_time=simulation_time,
        noise_type=noise_type,
        noise_intensity=noise_intensity,
        spatial_noise_correlation=spatial_correlation,
        temporal_noise_correlation=temporal_correlation,
    )

    # Run the full simulation as an ensemble, analyse it, and save the relevant things
    (
        ensemble_mean_power_spectra,
        linear_cross_correlation_mean,
        linear_cross_correlation_std,
    ) = run_ensemble_and_analyse(
        ensemble_number,
        simulation_parameters,
        SimulationType.FULL_SIMULATION,
        animation_save,
        path_to_save,
    )
    average_structure_function_array = analyse_spatial_wave_number(
        ensemble_mean_power_spectra, simulation_parameters, path_to_save
    )
    save_array(
        path_to_save,
        simulation_parameters,
        ensemble_mean_power_spectra,
        average_structure_function_array,
    )

    # Return the 'summary statistics' for this particular condition
    return np.array(
        [
            noise_type.name.lower(),
            noise_intensity,
            diffusion_type.name.lower(),
            diffusion_constant_xx,
            diffusion_constant_xy,
            diffusion_constant_yy,
            spatial_correlation,
            temporal_correlation,
            linear_cross_correlation_mean,
            linear_cross_correlation_std,
        ]
    )


def run_noise_in_parallel(
    noise_type: NoiseType,
    discretisation_parameters: DiscretisationParameters,
    simulation_time: float,
    noise_intensity: float,
    spatial_correlation: float,
    temporal_correlation: float,
    ensemble_number: int,
    path_to_save: pathlib.Path,
    animation_save: bool,
) -> np.ndarray:
    """
    Set up and do an ensemble run only for the noise for the given type, intensity, and correlation.
    Structured so taht this can be done via multiprocessing.

    Args:
        noise_type (NoiseType): the noise type for this condition
        discretisation_parameters (DiscretisationParameters): discretisation parameters for this
            condition
        simulation_time (float): simulation time for this condition
        noise_intensity (float): noise intensity for this condition
        spatial_correlation (float): For white noise, put 0. For correlated noise, the value for
            the spatial correlation of the noise
        temporal_correlation (float): For white noise, put 0. For correlated noise, the value for
            the temporal correlation of the noise
        ensemble_number (int): the number of runs considered an ensemble
        path_to_save (pathlib.Path): path to the main folder to save the data
        animation_save (bool): whether or not to save a representative animation of this condition

    Returns:
        np.ndarray: an array of the coherence metrics
    """
    # Set up the simulation parameters, but configured only for noise (so no FHN constants, no
    # diffusion constants). Use isotropic diffusion because it doesn't really matter either way
    simulation_parameters = SimulationParameters(
        fitzhugh_nagumo_constants=FitzHughNagumoConstants(0, 0, 0),
        discretisation_parameters=discretisation_parameters,
        diffusion_type=DiffusionType.ISOTROPIC,
        diffusion_constant=0.0,
        simulation_time=simulation_time,
        noise_type=noise_type,
        noise_intensity=noise_intensity,
        spatial_noise_correlation=spatial_correlation,
        temporal_noise_correlation=temporal_correlation,
    )

    # Run the simulation for noise only, analyse it, and save the relevant things
    (
        ensemble_mean_power_spectra,
        linear_cross_correlation_mean,
        linear_cross_correlation_std,
    ) = run_ensemble_and_analyse(
        ensemble_number,
        simulation_parameters,
        SimulationType.NOISE_ONLY,
        animation_save,
        path_to_save,
    )
    average_structure_function_array = analyse_spatial_wave_number(
        ensemble_mean_power_spectra, simulation_parameters, path_to_save
    )
    save_array(
        path_to_save,
        simulation_parameters,
        ensemble_mean_power_spectra,
        average_structure_function_array,
    )

    # Return the 'summary statistics' for this particular condition.
    return np.array(
        [
            noise_type.name.lower(),
            noise_intensity,
            spatial_correlation,
            temporal_correlation,
            linear_cross_correlation_mean,
            linear_cross_correlation_std,
        ]
    )


def run_metrics_batch(
    batch_parameter: ParametersBatch, data_path: pathlib.Path, animation_save: bool
) -> None:
    """Main function to run the metrics as a batch.

    Args:
        batch_parameter (ParametersBatch): parameters for this batch run
        data_path (pathlib.Path): where to save the data
        animation_save (bool): whether or not we save a representative animation of each condition
    """
    # Run for all types of diffusion and noise type
    for diffusion_type_string, noise_type_string in product(
        batch_parameter.diffusion_type_array, batch_parameter.noise_type_array
    ):
        # Make the folder to save this particular type of diffusion and noise type
        type_path = (
            data_path / f"{noise_type_string.lower()}_{diffusion_type_string.lower()}"
        )
        if not type_path.exists():
            type_path.mkdir()
        diffusion_type = DiffusionType.from_string(diffusion_type_string.upper())
        noise_type = NoiseType.from_string(noise_type_string.upper())

        # If isotropic, just use the xx_array as the main array for the diffusion constant, and use
        # 0 for everything else.
        if diffusion_type == DiffusionType.ISOTROPIC:
            diffusion_xx_array = batch_parameter.diffusion_constant_array
            diffusion_xy_array = [0]
            diffusion_yy_array = [0]
        elif diffusion_type == DiffusionType.ANISOTROPIC:
            diffusion_xx_array = batch_parameter.diffusion_constant_array
            diffusion_xy_array = batch_parameter.diffusion_constant_array
            diffusion_yy_array = batch_parameter.diffusion_constant_array
        else:
            # Should never get here, but to keep typecheck happy and as a fallback
            diffusion_xx_array = [0]
            diffusion_xy_array = [0]
            diffusion_yy_array = [0]

        if noise_type == NoiseType.CORRELATED:
            spatial_correlation_array = batch_parameter.spatial_correlation_array
            temporal_correlation_array = batch_parameter.temporal_correlation_array
        else:
            # If not correlated, it should be white noise, so use 0 for the correlation values
            spatial_correlation_array = [0]
            temporal_correlation_array = [0]

        # Prepare a path to save the resulting coherence metrics
        result_path = type_path / "coherence_metrics.csv"

        # Run for all the different correlation values
        for spatial_correlation, temporal_correlation in product(
            spatial_correlation_array, temporal_correlation_array
        ):
            # Make a folder for this particular correlation values
            correlated_path = (
                type_path
                / f"temporal_{temporal_correlation}_spatial_{spatial_correlation}"
            )
            if not correlated_path.exists():
                correlated_path.mkdir()

            # Run for the different diffusion constants
            for (
                diffusion_constant_xx,
                diffusion_constant_xy,
                diffusion_constant_yy,
            ) in product(diffusion_xx_array, diffusion_xy_array, diffusion_yy_array):
                # Use half of the available cpu count, or half of the number of noise intensity
                # whichever is smaller
                cpu_used = min(
                    int(cpu_count() / 2),
                    int(len(batch_parameter.noise_intensity_array) / 2) + 1,
                )

                # Run in parallel
                with Pool(processes=cpu_used) as pool:
                    results = pool.starmap(
                        run_ensemble_in_parallel,
                        zip(
                            repeat(diffusion_type),
                            repeat(noise_type),
                            repeat(batch_parameter.fitzhugh_nagumo_constants),
                            repeat(batch_parameter.discretisation_parameters),
                            repeat(diffusion_constant_xx),
                            repeat(diffusion_constant_xy),
                            repeat(diffusion_constant_yy),
                            repeat(batch_parameter.simulation_time),
                            batch_parameter.noise_intensity_array,
                            repeat(spatial_correlation),
                            repeat(temporal_correlation),
                            repeat(batch_parameter.ensemble_number),
                            repeat(correlated_path),
                            repeat(animation_save),
                        ),
                    )

                results_dataframe = pd.DataFrame(
                    results,
                    columns=[
                        "noise_type",
                        "noise_intensity",
                        "diffusion_type",
                        "diffusion_constant_xx",
                        "diffusion_constant_xy",
                        "diffusion_constant_yy",
                        "spatial_correlation",
                        "temporal_correlation",
                        "linear_cross_correlation_mean",
                        "linear_cross_correlation_std",
                    ],
                )

                # Save the coherence metrics once this particular correlation values are finished
                save_coherence_metrics(result_path, results_dataframe)


def run_metrics_noise(
    batch_parameters: ParametersBatch, data_path: pathlib.Path, animation_save: bool
) -> None:
    # Run for the different types of noise
    for noise_type_string in batch_parameters.noise_type_array:
        # Make a folder to save this type of noise
        noise_path = data_path / f"{noise_type_string.lower()}"
        if not noise_path.exists():
            noise_path.mkdir()
        noise_type = NoiseType.from_string(noise_type_string.upper())

        if noise_type == NoiseType.CORRELATED:
            spatial_correlation_array = batch_parameters.spatial_correlation_array
            temporal_correlation_array = batch_parameters.temporal_correlation_array
        else:
            # If not correlated, it should be white noise, so use 0 for the correlation values
            spatial_correlation_array = [0]
            temporal_correlation_array = [0]

        # Prepare a path to save the resulting coherence metrics
        result_path = noise_path / "coherence_metrics.csv"

        # Run for all the different correlation values
        for spatial_correlation, temporal_correlation in product(
            spatial_correlation_array, temporal_correlation_array
        ):
            # Make a folder for this particular correlation values
            correlated_path = (
                noise_path
                / f"temporal_{temporal_correlation}_spatial_{spatial_correlation}"
            )
            if not correlated_path.exists():
                correlated_path.mkdir()

            # Use half of the available cpu count, or half of the number of noise intensity
            # whichever is smaller
            cpu_used = min(
                (int(cpu_count() / 2)),
                int(len(batch_parameters.noise_intensity_array) / 2),
            )

            # Run in parallel
            with Pool(processes=cpu_used) as pool:
                results = pool.starmap(
                    run_noise_in_parallel,
                    zip(
                        repeat(noise_type),
                        repeat(batch_parameters.discretisation_parameters),
                        repeat(batch_parameters.simulation_time),
                        batch_parameters.noise_intensity_array,
                        repeat(spatial_correlation),
                        repeat(temporal_correlation),
                        repeat(batch_parameters.ensemble_number),
                        repeat(correlated_path),
                        repeat(animation_save),
                    ),
                )

            results_dataframe = pd.DataFrame(
                results,
                columns=[
                    "noise_type",
                    "noise_intensity",
                    "spatial_correlation",
                    "temporal_correlation",
                    "linear_cross_correlation_mean",
                    "linear_cross_correlation_std",
                ],
            )

            # Save the coherence metrics once this particular correlation values are finished
            save_coherence_metrics(result_path, results_dataframe)
