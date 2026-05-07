import typer
import pathlib
from typing import Annotated

from src.helper.animate import animate_plot, save_animation
from src.resonance_excitable_media.read_parameters import (
    read_parameters_single,
    ReadParameterError,
)
from src.resonance_excitable_media.run_single import run_single

app = typer.Typer(
    help="Run a single simulation with the given parameters. Default parameters available."
)

@app.command()
def default_path(
    animation_file_path: Annotated[
        pathlib.Path, typer.Option(help="The path to save the resulting animation.")
    ],
    noise_type: Annotated[
        str,
        typer.Option(
            help="Noise type for this run. Use 'white' for white noise "
            "and 'correlated' for correlated noise."
        ),
    ],
    diffusion_type: Annotated[
        str,
        typer.Option(
            help="Diffusion type for this run. Use 'isotropic' for isotropic diffusion "
            "and 'anisotropic for anisotropic diffusion."
        ),
    ],
) -> None:
    """Run a single simulation with default parameter values.

    Args:
        animation_file_name (pathlib.Path): The file name for the resulting animation.
        noise_type (str): Type of noise. Type 'white' for white noise, 'correlated' for correlated noise
        diffusion_type (str): Type of diffusion. Type 'isotropic' for isotropic diffusion,
            'anisotropic' for anisotropic diffusion
    """
    # Get the path for the default parameters depending on the type of noise and diffusion
    default_parent_parameters_path = (
        pathlib.Path(__file__).parent.parent.parent.resolve() / "docs" / "default_param"
    )
    simulation_constant_param_path = (
        default_parent_parameters_path / "default_param_simulation_constant.json"
    )
    diffusion_constant_param_path = (
        default_parent_parameters_path / f"default_param_{diffusion_type.lower()}.json"
    )
    noise_generation_param_path = (
        default_parent_parameters_path
        / f"default_param_{noise_type.lower()}_noise.json"
    )

    # Check that it's the right kind of diffusion and noise type. If not, then abort
    if not diffusion_constant_param_path.exists():
        print("Not the right kind of diffusion! Aborting")
        raise typer.Exit()

    if not noise_generation_param_path.exists():
        print("Not the right kind of noise type!")
        raise typer.Exit()

    # Run a single simulation, and then save to the provided path
    result = run_single(
        read_parameters_single(
            simulation_constant_param_path,
            diffusion_constant_param_path,
            noise_generation_param_path,
        ),
    )

    # Animate and save the plot
    anim = animate_plot(result.membrane_potential, result.membrane_potential.shape[2])
    # Save the animation to the given path
    save_animation(
        anim,
        animation_file_path,
    )


@app.command()
def own_path(
    animation_file_path: Annotated[
        pathlib.Path, typer.Option(help="The path to save the resulting animation.")
    ],
    simulation_constant_param_path: Annotated[
        pathlib.Path,
        typer.Option(
            help="Path to the JSON file for the simulation constants.",
        ),
    ],
    diffusion_constant_param_path: Annotated[
        pathlib.Path,
        typer.Option(help="Path to the JSON file for the diffusion constant."),
    ],
    noise_generation_param_path: Annotated[
        pathlib.Path,
        typer.Option(help="Path to the JSON file for the noise generation."),
    ],
) -> None:
    """Run a single simulation with user-provided (absolute) paths for the parameters."""

    try:
        simulation_parameters = read_parameters_single(
            simulation_constant_param_path,
            diffusion_constant_param_path,
            noise_generation_param_path,
        )
    except ReadParameterError:
        print("Unable to read parameters! Aborting!")
        raise typer.Exit()

    try:
        result = run_single(simulation_parameters)
    except ValueError:
        print("Infinite value encountered in time integration! Aborting!")
        raise typer.Exit()

    # Animate and save the plot
    anim = animate_plot(result.membrane_potential, result.membrane_potential.shape[2])
    # Save the animation to the given path
    save_animation(
        anim,
        animation_file_path,
    )

if __name__ == "__main__":
    app()
