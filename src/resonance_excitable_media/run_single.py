"""Script for running a single run."""

import numpy as np

from src.helper.datatypes import (
    ExcitableMedia,
    NoiseType,
    SimulationParameters,
)

from src.resonance_excitable_media.create_noise import (
    generate_spatiotemporal_correlated_noise,
    generate_white_noise,
)
from src.resonance_excitable_media.temporal_integration import diffusion_simulation


def run_single(simulation_parameters: SimulationParameters) -> ExcitableMedia:
    """
    Run a single simulation.

    Args:
        animation_file_name (pathlib.Path): name for the animation file
        simulation_parameters (SimulationParameters): simulation parameters to run the simulation

    Raises:
        ValueError: If within the temporal integration there is infinite value, raise again

    Returns:
        ExcitableMedia: the state of the excitable media after the simulation
    """
    # Calculate the number of timesteps required from the simulation time and the timestep
    number_of_time_steps = int(
        simulation_parameters.simulation_time
        / simulation_parameters.discretisation_parameters.temporal_step_size
    )
    timepoints = np.arange(
        0,
        simulation_parameters.simulation_time,
        simulation_parameters.discretisation_parameters.temporal_step_size,
    )

    # Generate the noise.
    # Grid size is the media size - 2 because the edges (boundaries) do not have noise for now,
    # as noise is only added in the derivatives and we are using a Neumann boundary condition
    if simulation_parameters.noise_type == NoiseType.WHITE:
        noise = generate_white_noise(
            noise_intensity=simulation_parameters.noise_intensity,
            grid_size=simulation_parameters.discretisation_parameters.grid_size[0] - 2,
            number_of_time_steps=number_of_time_steps,
        )
    elif simulation_parameters.noise_type == NoiseType.CORRELATED:
        noise = generate_spatiotemporal_correlated_noise(
            spatial_correlation=simulation_parameters.spatial_noise_correlation,
            temporal_correlation=simulation_parameters.temporal_noise_correlation,
            noise_intensity=simulation_parameters.noise_intensity,
            spatial_step_size=simulation_parameters.discretisation_parameters.spatial_step_size[
                0
            ],
            temporal_step_size=simulation_parameters.discretisation_parameters.temporal_step_size,
            grid_size=simulation_parameters.discretisation_parameters.grid_size[0] - 2,
            number_of_time_steps=number_of_time_steps,
        )
    else:
        # Should never get here to be honest, but put this here to keep typecheck happy
        noise = np.zeros(
            (
                simulation_parameters.discretisation_parameters.grid_size[0] - 2,
                simulation_parameters.discretisation_parameters.grid_size[1] - 2,
                number_of_time_steps,
            )
        )

    # Initiate the excitable media (for FitzHugh Nagumo definition)
    excitable_media = ExcitableMedia(
        grid_size=(
            simulation_parameters.discretisation_parameters.grid_size[0],
            simulation_parameters.discretisation_parameters.grid_size[1],
        ),
        temporal_size=number_of_time_steps,
    )
    try:
        # Run the diffusion
        return diffusion_simulation(
            simulation_parameters.fitzhugh_nagumo_constants,
            simulation_parameters.discretisation_parameters,
            timepoints,
            excitable_media,
            noise,
            simulation_parameters.diffusion_constant,
        )
    except ValueError:
        # If there is a ValueError within the diffusion_simulation, there is an infinite value,
        # so we raise it again to be resolved in the main function.
        raise
