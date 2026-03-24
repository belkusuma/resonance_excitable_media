"""Main script for running the simulation."""
import typer
import pathlib
from typing import Annotated
import json
import numpy as np
import matplotlib.pyplot as plt

from src.helper.datatypes import (
    FitzHughNagumoConstants,
    DiscretisationParameters,
    ExcitableMedia,
    SimulationParameters,
)
from src.resonance_excitable_media.create_noise import (
    generate_spatiotemporal_correlated_noise,
    generate_white_noise,
)
from src.resonance_excitable_media.temporal_integration import (
    isotropic_diffusion,
    anisotropic_diffusion,
)

from src.helper.animate import animate_plot, save_animation


def read_parameters(
    path: pathlib.Path,
) -> SimulationParameters:
    """Reading the parameters for running the simulation from the given json file.

    Args:
        path (pathlib.Path): path to the parameters json file

    Returns:
        SimulationParameters: a dataclass of all the parameters needed to run the simulation
    """
    with open(path) as f:
        parameters = json.load(f)

        fitzhugh_nagumo_constants = FitzHughNagumoConstants(
            a=parameters["fitzhugh_nagumo_constants"]["a"],
            b=parameters["fitzhugh_nagumo_constants"]["b"],
            epsilon=parameters["fitzhugh_nagumo_constants"]["epsilon"],
        )

        discretisation_parameters = DiscretisationParameters(
            grid_size=parameters["discretisation"]["grid_size"],
            spatial_step_size=parameters["discretisation"]["spatial_step_size"],
            temporal_step_size=parameters["discretisation"]["temporal_step_size"],
        )

        diffusion_constant = parameters["diffusion_constant"]
        noise_intensity = parameters["noise_intensity"]
        simulation_time = parameters["simulation_time"]

    return SimulationParameters(
        fitzhugh_nagumo_constants,
        discretisation_parameters,
        diffusion_constant,
        noise_intensity,
        simulation_time,
    )


def main(
    parameters_path: pathlib.Path = (
        pathlib.Path(__file__).parent.parent.resolve() / "default_param.json"
    ),
):
    simulation_parameters = read_parameters(parameters_path)

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
    # TODO: Make it configurable which noise we want
    noise = generate_white_noise(
        noise_intensity=simulation_parameters.noise_intensity,
        grid_size=simulation_parameters.discretisation_parameters.grid_size[0] - 2,
        number_of_time_steps=number_of_time_steps,
    )

    # Initiate the excitable media (for FitzHugh Nagumo definition)
    # TODO: Make configurable depending on the definition of an 'excitable media'
    excitable_media = ExcitableMedia(
        membrane_potential=np.zeros(
            (
                simulation_parameters.discretisation_parameters.grid_size[0],
                simulation_parameters.discretisation_parameters.grid_size[1],
                number_of_time_steps,
            )
        ),
        potassium_conductance=np.zeros(
            (
                simulation_parameters.discretisation_parameters.grid_size[0],
                simulation_parameters.discretisation_parameters.grid_size[1],
                number_of_time_steps,
            )
        ),
    )

    # Run the diffusion
    # TODO: Make configurable depending on the type of diffusions
    result = isotropic_diffusion(
        simulation_parameters.fitzhugh_nagumo_constants,
        simulation_parameters.discretisation_parameters,
        timepoints,
        excitable_media,
        noise,
        simulation_parameters.diffusion_constant,
    )

    # Animate and save the plot
    anim = animate_plot(result.membrane_potential, timepoints)
    plt.show()


if __name__ == "__main__":
    typer.run(main)
