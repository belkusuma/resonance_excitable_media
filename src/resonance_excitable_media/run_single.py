"""Script for running a single run and saving a video of the result."""

import typer
import pathlib
from typing import Annotated
import json
import numpy as np
import sys

from src.helper.datatypes import (
    FitzHughNagumoConstants,
    DiscretisationParameters,
    ExcitableMedia,
    SimulationParameters,
    NoiseType,
    DiffusionType,
    DiffusionTensor,
)
from src.helper.validate_param import validate_json_schema
from src.helper.diffusion_tensor_mask import (
    import_mask_of_tract,
    create_diffusion_tensor_from_mask,
)
from src.helper.animate import animate_plot, save_animation
from src.resonance_excitable_media.create_noise import (
    generate_spatiotemporal_correlated_noise,
    generate_white_noise,
)
from src.resonance_excitable_media.temporal_integration import (
    isotropic_diffusion,
    anisotropic_diffusion,
)


def read_parameters(
    path: pathlib.Path, noise_type: NoiseType, diffusion_type: DiffusionType
) -> SimulationParameters:
    """Reading the parameters for running the simulation from the given json file.

    Args:
        path (pathlib.Path): path to the parameters json file
        noise_type (NoiseType): type of supported noise generation
        diffusion_type (DiffusionType): type of supported diffusion

    Returns:
        SimulationParameters: a dataclass of all the parameters needed to run the simulation
    """
    # Find the schema for the particular noise and diffusion type, and then validate
    schema_path = (
        pathlib.Path(__file__).parent.parent.parent.resolve()
        / "docs"
        / "param_schema"
        / f"{noise_type.name.lower()}_noise_{diffusion_type.name.lower()}.schema.json"
    )
    validate_json_schema(path, schema_path)

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
        simulation_time = parameters["simulation_time"]
        noise_intensity = parameters["noise_intensity"]

        if diffusion_type == DiffusionType.ISOTROPIC:
            diffusion_constant = parameters["diffusion_constant"]
        elif diffusion_type == DiffusionType.ANISOTROPIC:
            # Make sure that the number of masks is the same as the number of diffusion values given
            assert len(parameters["anisotropic_diffusion_tensor"]["mask_paths"]) == len(
                parameters["anisotropic_diffusion_tensor"]["diffusion_values"]
            )

            mask = []
            for i in range(
                len(parameters["anisotropic_diffusion_tensor"]["mask_paths"])
            ):
                # The mask has to be in the same folder as the parameter json file.
                # Can't really figure out how else to do this...
                mask_path = (
                    path.parent.resolve()
                    / parameters["anisotropic_diffusion_tensor"]["mask_paths"][i]
                )
                individual_mask = import_mask_of_tract(mask_path)

                # Make sure that the size of the mask is the same as the grid size
                assert (
                    individual_mask.shape[0] == discretisation_parameters.grid_size[0]
                )
                assert (
                    individual_mask.shape[1] == discretisation_parameters.grid_size[1]
                )
                mask.append(individual_mask)

            diffusion_constant = create_diffusion_tensor_from_mask(
                d_isotropic=parameters["isotropic_diffusion_constant"],
                d_anisotropic=parameters["anisotropic_diffusion_tensor"][
                    "diffusion_values"
                ],
                mask=mask,
            )

        if noise_type == NoiseType.WHITE:
            return SimulationParameters(
                fitzhugh_nagumo_constants=fitzhugh_nagumo_constants,
                discretisation_parameters=discretisation_parameters,
                diffusion_constant=diffusion_constant,
                simulation_time=simulation_time,
                noise_intensity=noise_intensity,
            )
        elif noise_type == NoiseType.CORRELATED:
            spatial_noise_correlation = parameters["noise_correlation"]["spatial"]
            temporal_noise_correlation = parameters["noise_correlation"]["temporal"]
            return SimulationParameters(
                fitzhugh_nagumo_constants=fitzhugh_nagumo_constants,
                discretisation_parameters=discretisation_parameters,
                diffusion_constant=diffusion_constant,
                simulation_time=simulation_time,
                noise_intensity=noise_intensity,
                spatial_noise_correlation=spatial_noise_correlation,
                temporal_noise_correlation=temporal_noise_correlation,
            )


def main(
    noise_type: Annotated[
        str,
        typer.Argument(
            help="Type of noise. Type 'white' for white noise, 'correlated' for correlated noise"
        ),
    ],
    diffusion_type: Annotated[
        str,
        typer.Argument(
            help="Type of diffusion. Type 'isotropic' for isotropic diffusion, "
            "'anisotropic' for anisotropic diffusion"
        ),
    ],
    animation_file_name: Annotated[
        pathlib.Path, typer.Argument(help="The file name for the resulting animation.")
    ],
    default_param: Annotated[
        bool,
        typer.Option(help="Whether or not to use the default parameters in the repo"),
    ] = True,
    own_param_path: Annotated[
        pathlib.Path | None,
        typer.Option(
            help="If not using default, provide an (absolute) path for a valid JSON parameter file"
        ),
    ] = None,
):
    """Main function to run the simulation."""
    # Check whether the noise type is supported. If not, abort
    if noise_type.lower() == "white":
        noise_type_enum = NoiseType.WHITE
    elif noise_type.lower() == "correlated":
        noise_type_enum = NoiseType.CORRELATED
    else:
        print("Not a supported noise type!")
        sys.exit()

    # Check whether the diffusion type is supported. If not, abort
    if diffusion_type.lower() == "isotropic":
        diffusion_type_enum = DiffusionType.ISOTROPIC
    elif diffusion_type.lower() == "anisotropic":
        diffusion_type_enum = DiffusionType.ANISOTROPIC
    else:
        print("Not a supported diffusion type!")
        sys.exit()

    # If using default parameters, retrieve the default parameter file from the repo
    if default_param:
        parameters_path = (
            pathlib.Path(__file__).parent.parent.parent.resolve()
            / "docs"
            / "default_param"
            / f"default_param_{noise_type}_noise_{diffusion_type}.json"
        )
    else:
        # Else, make sure that the user has given a path to a parameter file
        if own_param_path:
            parameters_path = own_param_path
        else:
            print("Path to parameters file required!")
            sys.exit()

    # Read (and validate) the parameter file
    simulation_parameters = read_parameters(
        parameters_path, noise_type_enum, diffusion_type_enum
    )

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
    if noise_type_enum == NoiseType.WHITE:
        noise = generate_white_noise(
            noise_intensity=simulation_parameters.noise_intensity,
            grid_size=simulation_parameters.discretisation_parameters.grid_size[0] - 2,
            number_of_time_steps=number_of_time_steps,
        )
    elif noise_type_enum == NoiseType.CORRELATED:
        # Make sure that if the noise type is correlated, we have the correlation values
        assert simulation_parameters.spatial_noise_correlation
        assert simulation_parameters.temporal_noise_correlation

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

    # Initiate the excitable media (for FitzHugh Nagumo definition)
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
    if diffusion_type_enum == DiffusionType.ISOTROPIC:
        # For isotropic diffusion, only one value for the diffusion constant, thus a float
        assert isinstance(simulation_parameters.diffusion_constant, float)
        result = isotropic_diffusion(
            simulation_parameters.fitzhugh_nagumo_constants,
            simulation_parameters.discretisation_parameters,
            timepoints,
            excitable_media,
            noise,  # type: ignore
            simulation_parameters.diffusion_constant,
        )
    elif diffusion_type_enum == DiffusionType.ANISOTROPIC:
        # For anisotropic diffusion, each square in the grid will have 3 values of diffusion, so
        # need to be a tensor
        assert isinstance(simulation_parameters.diffusion_constant, DiffusionTensor)
        result = anisotropic_diffusion(
            simulation_parameters.fitzhugh_nagumo_constants,
            simulation_parameters.discretisation_parameters,
            timepoints,
            excitable_media,
            noise,  # type: ignore
            simulation_parameters.diffusion_constant,
        )

    # Animate and save the plot
    anim = animate_plot(result.membrane_potential, timepoints)  # type: ignore
    # Save the animation to the data folder within the repo. User should move the gifs themselves if
    # needed/wanted
    save_animation(
        anim,
        pathlib.Path(__file__).parent.parent.parent.resolve()
        / "data"
        / animation_file_name,
    )


if __name__ == "__main__":
    typer.run(main)
