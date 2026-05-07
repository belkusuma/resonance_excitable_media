"""Functions to read the parameters JSON files accordingly."""

import numpy as np
import pathlib
import json
from src.helper.datatypes import (
    NoiseType,
    DiffusionType,
    SimulationParameters,
    FitzHughNagumoConstants,
    DiscretisationParameters,
    ParametersBatch,
    DiffusionTensor,
)
from src.helper.validate_param import validate_json_schema
from src.helper.diffusion_tensor_mask import (
    import_mask_of_tract,
    create_diffusion_tensor_from_mask,
)
from jsonschema.exceptions import ValidationError


class ReadParameterError(Exception):
    """Error specifically when failed to read parameters."""

    def __init__(self):
        super().__init__()


def read_parameters_single(
    simulation_constant_param_path: pathlib.Path,
    diffusion_constant_param_path: pathlib.Path,
    noise_generation_param_path: pathlib.Path,
) -> SimulationParameters:
    """Reading the parameters for running the simulation from the given json file.

    Args:
        path (pathlib.Path): path to the parameters json file
        noise_type (NoiseType): type of supported noise generation
        diffusion_type (DiffusionType): type of supported diffusion

    Raises:
        ReadParameterError: when parameters are not read properly

    Returns:
        SimulationParameters: a dataclass of all the parameters needed to run the simulation
    """
    # Find the schema for the particular noise and diffusion type, and then validate
    schema_parent_path = (
        pathlib.Path(__file__).parent.parent.parent.resolve() / "docs" / "param_schema"
    )

    simulation_constant_schema_path = (
        schema_parent_path / "simulation_constant.schema.json"
    )
    diffusion_constant_schema_path = (
        schema_parent_path / "diffusion_constant.schema.json"
    )
    noise_generation_schema_path = schema_parent_path / "noise_generation.schema.json"

    try:
        validate_json_schema(
            simulation_constant_param_path, simulation_constant_schema_path
        )
        validate_json_schema(
            diffusion_constant_param_path, diffusion_constant_schema_path
        )
        validate_json_schema(noise_generation_param_path, noise_generation_schema_path)

        fitzhugh_nagumo_constants, discretisation_parameters, simulation_time, _ = (
            read_simulation_constant_parameters(simulation_constant_param_path)
        )
        diffusion_type, diffusion_constant = read_diffusion_constant_parameters(
            diffusion_constant_param_path, discretisation_parameters
        )
        (
            noise_type,
            noise_intensity,
            spatial_noise_correlation,
            temporal_noise_correlation,
        ) = read_noise_generation_parameters(noise_generation_param_path)

        return SimulationParameters(
            fitzhugh_nagumo_constants=fitzhugh_nagumo_constants,
            discretisation_parameters=discretisation_parameters,
            diffusion_type=diffusion_type,
            diffusion_constant=diffusion_constant,
            simulation_time=simulation_time,
            noise_type=noise_type,
            noise_intensity=noise_intensity,
            spatial_noise_correlation=spatial_noise_correlation,
            temporal_noise_correlation=temporal_noise_correlation,
        )
    except (ValidationError, NotImplementedError, KeyError, AttributeError):
        raise ReadParameterError()


def read_parameters_batch(
    simulation_constant_param_path: pathlib.Path,
    noise_and_diffusion_batch_path: pathlib.Path,
) -> ParametersBatch:
    """Reading the parameters for running batch simulations (i.e. for metric calculations) from a
    given JSON file.

    Args:
        simulation_constant_param_path (pathlib.Path): path to the simulation constant json file
        noise_and_diffusion_batch_path (pathlib.Path): path to the combinations of noise and
            diffusion constants

    Raises:
        ReadParameterError: when parameters are not read properly

    Returns:
        ParametersBatch: a dataclass of the batch parameters
    """
    # Find the schema for the particular noise and diffusion type, and then validate
    schema_parent_path = (
        pathlib.Path(__file__).parent.parent.parent.resolve() / "docs" / "param_schema"
    )

    simulation_constant_schema_path = (
        schema_parent_path / "simulation_constant.schema.json"
    )
    batch_schema_path = schema_parent_path / "batch_param.schema.json"

    try:
        validate_json_schema(
            simulation_constant_param_path, simulation_constant_schema_path
        )
        validate_json_schema(noise_and_diffusion_batch_path, batch_schema_path)
        (
            fitzhugh_nagumo_constants,
            discretisation_parameters,
            simulation_time,
            ensemble_number,
        ) = read_simulation_constant_parameters(simulation_constant_param_path)

        (
            diffusion_type_array,
            diffusion_xx_array,
            diffusion_xy_array,
            diffusion_yy_array,
            noise_type_array,
            noise_intensity_array,
            spatial_correlation_array,
            temporal_correlation_array,
            cross_correlation_direction
        ) = read_noise_and_diffusion_combinations(noise_and_diffusion_batch_path)

        return ParametersBatch(
            fitzhugh_nagumo_constants=fitzhugh_nagumo_constants,
            discretisation_parameters=discretisation_parameters,
            simulation_time=simulation_time,
            ensemble_number=ensemble_number,
            diffusion_type_array=diffusion_type_array,
            diffusion_xx_array=diffusion_xx_array,
            diffusion_xy_array=diffusion_xy_array,
            diffusion_yy_array=diffusion_yy_array,
            noise_type_array=noise_type_array,
            noise_intensity_array=noise_intensity_array,
            spatial_correlation_array=spatial_correlation_array,
            temporal_correlation_array=temporal_correlation_array,
            cross_correlation_direction=cross_correlation_direction
        )

    except (ValidationError, AssertionError):
        raise ReadParameterError()


def read_simulation_constant_parameters(
    simulation_constant_param_path: pathlib.Path,
) -> tuple[FitzHughNagumoConstants, DiscretisationParameters, float, int]:
    """Reads the simulation constant parameter file. Assumes file is valid.

    Args:
        simulation_constant_param_path (pathlib.Path): path to the simulation constant file

    Returns:
        FitzHughNagumoConstants: the constants for the FHN equations
        DiscretisationParameters: the discretisation parameters
        float: how long the simulation should run for
        int: number of ensembles to run
    """
    with open(simulation_constant_param_path) as simulation_constant_file:
        simulation_constant_parameters = json.load(simulation_constant_file)

        fitzhugh_nagumo_constants = FitzHughNagumoConstants(
            a=simulation_constant_parameters["fitzhugh_nagumo_constants"]["a"],
            b=simulation_constant_parameters["fitzhugh_nagumo_constants"]["b"],
            epsilon=simulation_constant_parameters["fitzhugh_nagumo_constants"][
                "epsilon"
            ],
        )

        discretisation_parameters = DiscretisationParameters(
            grid_size=simulation_constant_parameters["discretisation"]["grid_size"],
            spatial_step_size=simulation_constant_parameters["discretisation"][
                "spatial_step_size"
            ],
            temporal_step_size=simulation_constant_parameters["discretisation"][
                "temporal_step_size"
            ],
        )
        simulation_time = simulation_constant_parameters["simulation_time"]

        if "ensemble_number" in simulation_constant_parameters:
            ensemble_number = simulation_constant_parameters["ensemble_number"]
        else:
            ensemble_number = 1

    return (
        fitzhugh_nagumo_constants,
        discretisation_parameters,
        simulation_time,
        ensemble_number,
    )


def read_diffusion_constant_parameters(
    diffusion_constant_param_path: pathlib.Path,
    discretisation_parameters: DiscretisationParameters,
) -> tuple[DiffusionType, float | DiffusionTensor]:
    """Reads the diffusion constant parameters. Assumes file is valid.

    Args:
        diffusion_constant_param_path (pathlib.Path): path to the diffusion constant file
        discretisation_parameters (DiscretisationParameters): the discretisation parameters

    Raises:
        KeyError: If diffusion is set to be anisotropic but no anisotropic diffusion constant
        NotImplementedError: If diffusion ise set to be other than isotropic or anisotropic

    Returns:
        diffusion_type: the type of diffusion
        diffusion_constant: either float or a tensor
    """
    with open(diffusion_constant_param_path) as diffusion_constant_file:
        diffusion_constant_parameters = json.load(diffusion_constant_file)
        diffusion_type = DiffusionType.from_string(
            diffusion_constant_parameters["diffusion_type"].upper()
        )

        if diffusion_type == DiffusionType.ISOTROPIC:
            diffusion_constant = diffusion_constant_parameters[
                "isotropic_diffusion_constant"
            ]

        elif diffusion_type == DiffusionType.ANISOTROPIC:
            if not "anisotropic_diffusion_tensor" in diffusion_constant_parameters:
                print(
                    "Anisotropic diffusion type must have anisotropic_diffusion_tensor!"
                )
                raise KeyError

            # Make sure that the number of masks is the same as the number of diffusion values given
            assert len(
                diffusion_constant_parameters["anisotropic_diffusion_tensor"][
                    "mask_paths"
                ]
            ) == len(
                diffusion_constant_parameters["anisotropic_diffusion_tensor"][
                    "diffusion_values"
                ]
            )

            tract_mask = []
            for i in range(
                len(
                    diffusion_constant_parameters["anisotropic_diffusion_tensor"][
                        "mask_paths"
                    ]
                )
            ):
                # The mask has to be in the same folder as the parameter json file.
                # Can't really figure out how else to do this...
                mask_path = (
                    diffusion_constant_param_path.parent.resolve()
                    / diffusion_constant_parameters["anisotropic_diffusion_tensor"][
                        "mask_paths"
                    ][i]
                )

                # If the mask path is not an image, make the entire thing the tract
                if mask_path.suffix in [".png", ".jpg", ".jpeg"]:
                    individual_mask = import_mask_of_tract(mask_path)
                    # Make sure that the size of the mask is the same as the grid size
                    assert (
                        individual_mask.shape[0]
                        == discretisation_parameters.grid_size[0]
                    )
                    assert (
                        individual_mask.shape[1]
                        == discretisation_parameters.grid_size[1]
                    )
                else:
                    individual_mask = np.ones(
                        discretisation_parameters.grid_size, dtype=bool
                    )

                tract_mask.append(individual_mask)

            diffusion_constant = create_diffusion_tensor_from_mask(
                d_isotropic=diffusion_constant_parameters[
                    "isotropic_diffusion_constant"
                ],
                d_anisotropic=diffusion_constant_parameters[
                    "anisotropic_diffusion_tensor"
                ]["diffusion_values"],
                mask=tract_mask,
            )
        else:
            print("This diffusion type is not yet implemented!")
            raise NotImplementedError

    return diffusion_type, diffusion_constant


def read_noise_generation_parameters(
    noise_generation_param_path: pathlib.Path,
) -> tuple[NoiseType, float, float, float]:
    """Reads the noise generation parameter file. Assumes file is valid.

    Args:
        noise_generation_param_path (pathlib.Path): path to the noise generation file

    Raises:
        KeyError: If noise is set to be correlated but no correlation is found.
        NotImplementedError: If noise not white or correlated

    Returns:
        noise_type: the type of noise
        noise_intensity: intensity of the noise
        spatial_noise_correlation: the spatial correlation value
        temporal_noise_correlation: the temporal correlation value
    """
    with open(noise_generation_param_path) as noise_generation_file:
        noise_generation_parameters = json.load(noise_generation_file)
        noise_type = NoiseType.from_string(
            noise_generation_parameters["noise_type"].upper()
        )
        noise_intensity = noise_generation_parameters["noise_intensity"]

        if noise_type == NoiseType.WHITE:
            spatial_noise_correlation = 0
            temporal_noise_correlation = 0
        elif noise_type == NoiseType.CORRELATED:
            if "noise_correlation" in noise_generation_parameters:
                spatial_noise_correlation = noise_generation_parameters[
                    "noise_correlation"
                ]["spatial"]
                temporal_noise_correlation = noise_generation_parameters[
                    "noise_correlation"
                ]["temporal"]
            else:
                print("Correlated noise must have noise_correlation!")
                raise KeyError
        else:
            print("This noise type is not yet implemented!")
            raise NotImplementedError

    return (
        noise_type,
        noise_intensity,
        spatial_noise_correlation,
        temporal_noise_correlation,
    )


def read_noise_and_diffusion_combinations(
    noise_and_diffusion_batch_path: pathlib.Path,
) -> tuple[
    list[str],
    list[float],
    list[float],
    list[float],
    list[str],
    list[float],
    list[float],
    list[float],
    np.ndarray | None
]:
    """Reads the noise and diffusion combination file for running batch simulations.

    Args:
        noise_and_diffusion_batch_path (pathlib.Path): path to the file

    Returns:
        diffusion_type_array (list[str]): a list of all the diffusion type
        diffusion_xx_array (list[float]): a list of all the diffusion constant in xx. If isotropic
            this should be the one filled for diffusion constant.
        diffusion_xy_array (list[float]): a list of all the diffusion constant in xy.
        diffusion_yy_array (list[float]): a list of all the diffusion constant in yy.
        noise_type_array (list[str]): a list of all the noise type
        noise_intensity_array (list[float]): a list of all the noise intensities
        spatial_correlation_array (list[float]): a list of all the spatial correlation value. 
            If white, should fill with 0.
        temporal_correlation_array (list[float]): a list of all the temporal correlation value. 
            If white, should fill with 0.
    """
    with open(noise_and_diffusion_batch_path) as noise_and_diffusion_file:
        parameters = json.load(noise_and_diffusion_file)

        diffusion_type_array = parameters["diffusion_type_array"]
        diffusion_xx_array = parameters["diffusion_xx_array"]
        diffusion_xy_array = parameters["diffusion_xy_array"]
        diffusion_yy_array = parameters["diffusion_yy_array"]

        assert len(diffusion_xx_array) == len(diffusion_xy_array)
        assert len(diffusion_xy_array) == len(diffusion_yy_array)

        noise_type_array = parameters["noise_type_array"]
        noise_intensity_array = parameters["noise_intensity_array"]
        spatial_correlation_array = parameters["spatial_correlation_array"]
        temporal_correlation_array = parameters["temporal_correlation_array"]
        
        if "cross_correlation_direction" in parameters:
            cross_correlation_direction = np.asarray(parameters["cross_correlation_direction"])
        else:
            cross_correlation_direction = None


    return (
        diffusion_type_array,
        diffusion_xx_array,
        diffusion_xy_array,
        diffusion_yy_array,
        noise_type_array,
        noise_intensity_array,
        spatial_correlation_array,
        temporal_correlation_array,
        cross_correlation_direction,
    )
