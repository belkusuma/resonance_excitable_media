import typer
from typing import Annotated
import pathlib

app = typer.Typer(help="Run batch simulations with the given parameters.")

from src.resonance_excitable_media.read_parameters import (
    read_parameters_batch,
)
from src.resonance_excitable_media.run_metric import (
    run_metrics_noise,
    run_metrics_batch,
)


@app.command()
def own_path(
    results_path: Annotated[
        pathlib.Path, typer.Option(help="The path to a folder to save the results in.")
    ],
    simulation_constant_param_path: Annotated[
        pathlib.Path,
        typer.Option(help="Path to the JSON file for the simulation constants."),
    ],
    batch_parameter_path: Annotated[
        pathlib.Path,
        typer.Option(
            help="Path to the JSON file for the batch conditions for running this multiple run."
        ),
    ],
    animation_save: Annotated[
        bool,
        typer.Option(
            help="To save a random run of the ensemble as representative animation.",
        ),
    ] = False,
):
    """Run metrics for batch parameters with user-provided (absolute) paths for the parameters."""

    # Read the parameters for this batch run
    batch_parameters = read_parameters_batch(
        simulation_constant_param_path, batch_parameter_path
    )

    # Run the metrics for noise only, and then for the batch run
    run_metrics_noise(batch_parameters, results_path, animation_save)
    run_metrics_batch(batch_parameters, results_path, animation_save)


if __name__ == "__main__":
    app()
