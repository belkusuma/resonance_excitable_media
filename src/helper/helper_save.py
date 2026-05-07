import pathlib
import pandas as pd
from src.helper.datatypes import SimulationParameters, DiffusionType, DiffusionTensor


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

def save_dataframe(path_to_save: pathlib.Path, dataframe: pd.DataFrame) -> None:
    """Save the coherence metrics to file. If the file already exists, then append.

    Args:
        path_to_save (pathlib.Path): path to the file to save the coherence metrics.
        dataframe (pd.DataFrame): the coherence metrics as a pandas dataframe
    """
    if path_to_save.exists():
        dataframe.to_csv(path_to_save, mode="a", index=False, header=False)
    else:
        dataframe.to_csv(path_to_save, mode="w", index=False, header=True)

