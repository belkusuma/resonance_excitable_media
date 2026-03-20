import numpy as np

from src.helper.datatypes import (
    FitzHughNagumoConstants,
    DiscretisationParameters,
    ExcitableMedia,
    DiffusionTensor,
)

def fitzhugh_nagumo_equation(
    fitzhugh_nagumo_constants: FitzHughNagumoConstants,
    membrane_potential,
    potassium_conductance,
):
    # Calculate the FitzHugh-Nagumo equations for all elements in the grid, except for the
    # boundaries
    # Modify for well-behaved behaviour according to Perc, 2005
    f = (
        (1 / fitzhugh_nagumo_constants.epsilon)
        * membrane_potential[1:-1, 1:-1]
        * (1 - membrane_potential[1:-1, 1:-1])
        * (
            membrane_potential[1:-1, 1:-1]
            - (
                (potassium_conductance[1:-1, 1:-1] + fitzhugh_nagumo_constants.b)
                / fitzhugh_nagumo_constants.a
            )
        )
    )
    f_uv = np.where(membrane_potential[1:-1, 1:-1] <= 1, f, -np.abs(f))

    g = membrane_potential[1:-1, 1:-1] - potassium_conductance[1:-1, 1:-1]
    g_uv = np.where(potassium_conductance[1:-1, 1:-1] >= 0, g, np.abs(g))

    return f_uv, g_uv


def temporal_integration(
    new_membrane_potential: np.ndarray,
    old_membrane_potential: np.ndarray,
    new_potassium_conductance: np.ndarray,
    old_potassium_conductance: np.ndarray,
    temporal_step_size: float,
    f_uv: np.ndarray,
    g_uv: np.ndarray,
    diffusion_term: np.ndarray,
    noise: np.ndarray,
):
    new_membrane_potential[1:-1, 1:-1] = (
        f_uv + diffusion_term + noise
    ) * temporal_step_size + old_membrane_potential[1:-1, 1:-1]
    new_potassium_conductance[1:-1, 1:-1] = (
        g_uv
    ) * temporal_step_size + old_potassium_conductance[1:-1, 1:-1]


def neumann_boundary_condition(
    membrane_potential: np.ndarray, potassium_conductance: np.ndarray
):
    membrane_potential[0, :] = membrane_potential[1, :]
    potassium_conductance[0, :] = potassium_conductance[1, :]

    membrane_potential[-1, :] = membrane_potential[-2, :]
    potassium_conductance[-1, :] = potassium_conductance[-2, :]

    membrane_potential[:, 0] = membrane_potential[:, 1]
    potassium_conductance[:, 0] = potassium_conductance[:, 1]

    membrane_potential[:, -1] = membrane_potential[:, -2]
    potassium_conductance[:, -1] = potassium_conductance[:, -2]


def anisotropic_diffusion(
    fitzhugh_nagumo_constants: FitzHughNagumoConstants,
    discretisation_parameters: DiscretisationParameters,
    timepoints: np.ndarray,
    excitable_media: ExcitableMedia,
    noise: np.ndarray,
    diffusion_tensor: DiffusionTensor,
) -> ExcitableMedia:
    # TODO: CHECK SIZES ON THE ARRAYS

    for t in range(len(timepoints) - 1):
        # Calculate the FitzHugh-Nagumo equations for all elements in the grid, except for the
        # boundaries
        # Modify for well-behaved behaviour according to Perc, 2005

        f_uv, g_uv = fitzhugh_nagumo_equation(
            fitzhugh_nagumo_constants,
            excitable_media.membrane_potential[:, :, t],
            excitable_media.potassium_conductance[:, :, t],
        )

        # Calculate the second-order spatial differentiation with discretisation
        u_xx = (
            excitable_media.membrane_potential[2:, 1:-1, t]
            + excitable_media.membrane_potential[:-2, 1:-1, t]
            - 2 * excitable_media.membrane_potential[1:-1, 1:-1, t]
        ) / (discretisation_parameters.spatial_step_size[0] ** 2)
        u_yy = (
            excitable_media.membrane_potential[1:-1, 2:, t]
            + excitable_media.membrane_potential[1:-1, :-2, t]
            - 2 * excitable_media.membrane_potential[1:-1, 1:-1, t]
        ) / (discretisation_parameters.spatial_step_size[1] ** 2)
        u_xy = (
            excitable_media.membrane_potential[2:, 2:, t]
            + excitable_media.membrane_potential[:-2, :-2, t]
            - excitable_media.membrane_potential[2:, :-2, t]
            - excitable_media.membrane_potential[:-2, 2:, t]
        ) / (
            4
            * discretisation_parameters.spatial_step_size[0]
            * discretisation_parameters.spatial_step_size[1]
        )

        # Calculate the diffusion term
        diffusion_term = (
            diffusion_tensor.d_xx[1:-1, 1:-1] * u_xx
            + diffusion_tensor.d_yy[1:-1, 1:-1] * u_yy
            + 2 * diffusion_tensor.d_xy[1:-1, 1:-1] * u_xy
        )

        # Do temporal integration to get to the next time-step (for the elements that are not on
        # the boundary)
        temporal_integration(
            excitable_media.membrane_potential[:, :, t + 1],
            excitable_media.membrane_potential[:, :, t],
            excitable_media.potassium_conductance[:, :, t + 1],
            excitable_media.potassium_conductance[:, :, t],
            discretisation_parameters.temporal_step_size,
            f_uv,
            g_uv,
            diffusion_term,
            noise[:, :, t],
        )

        # Use Neumann boundary conditions (spatial differentiation = 0) for the elements on the
        # boundary
        neumann_boundary_condition(
            excitable_media.membrane_potential[:, :, t + 1],
            excitable_media.potassium_conductance[:, :, t + 1],
        )

    return excitable_media


def isotropic_diffusion(
    fitzhugh_nagumo_constants: FitzHughNagumoConstants,
    discretisation_parameters: DiscretisationParameters,
    timepoints: np.ndarray,
    excitable_media: ExcitableMedia,
    noise: np.ndarray,
    diffusion_constant: float,
) -> ExcitableMedia:
    # TODO: CHECK SIZES ON THE ARRAYS

    for t in range(len(timepoints) - 1):
        # Calculate the FitzHugh-Nagumo equations for all elements in the grid, except for the
        # boundaries
        # Modify for well-behaved behaviour according to Perc, 2005

        f_uv, g_uv = fitzhugh_nagumo_equation(
            fitzhugh_nagumo_constants,
            excitable_media.membrane_potential[:, :, t],
            excitable_media.potassium_conductance[:, :, t],
        )

        # Calculate the second-order spatial differentiation with discretisation
        u_xx = (
            excitable_media.membrane_potential[2:, 1:-1, t]
            + excitable_media.membrane_potential[:-2, 1:-1, t]
            - 2 * excitable_media.membrane_potential[1:-1, 1:-1, t]
        ) / (discretisation_parameters.spatial_step_size[0] ** 2)
        u_yy = (
            excitable_media.membrane_potential[1:-1, 2:, t]
            + excitable_media.membrane_potential[1:-1, :-2, t]
            - 2 * excitable_media.membrane_potential[1:-1, 1:-1, t]
        ) / (discretisation_parameters.spatial_step_size[1] ** 2)

        # Calculate the diffusion term
        diffusion_term = diffusion_constant * (u_xx + u_yy)

        # Do temporal integration to get to the next time-step (for the elements that are not on
        # the boundary)
        temporal_integration(
            excitable_media.membrane_potential[:, :, t + 1],
            excitable_media.membrane_potential[:, :, t],
            excitable_media.potassium_conductance[:, :, t + 1],
            excitable_media.potassium_conductance[:, :, t],
            discretisation_parameters.temporal_step_size,
            f_uv,
            g_uv,
            diffusion_term,
            noise[:, :, t],
        )

        # Use Neumann boundary conditions (spatial differentiation = 0) for the elements on the
        # boundary
        neumann_boundary_condition(
            excitable_media.membrane_potential[:, :, t + 1],
            excitable_media.potassium_conductance[:, :, t + 1],
        )

    return excitable_media
