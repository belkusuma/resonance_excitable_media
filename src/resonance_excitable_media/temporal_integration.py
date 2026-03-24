"""Temporal integration of the excitable media. 

Modification of FitzHugh Nagumo and isotropic diffusion done according to 
{
M. Perc, ‘Spatial coherence resonance in excitable media’, 
Phys. Rev. E, vol. 72, no. 1, p. 016207, Jul. 2005, 
doi: 10.1103/PhysRevE.72.016207.
}

Anisotropic diffusion is own derivation.
"""
import numpy as np

from src.helper.datatypes import (
    FitzHughNagumoConstants,
    DiscretisationParameters,
    ExcitableMedia,
    DiffusionTensor,
)

def fitzhugh_nagumo_equation(
    fitzhugh_nagumo_constants: FitzHughNagumoConstants,
    membrane_potential: np.ndarray,
    potassium_conductance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the FitzHugh Nagumo equation at each timestep.

    Args:
        fitzhugh_nagumo_constants (FitzHughNagumoConstants): the constants for FHN
        membrane_potential (np.ndarray): membrane potential (u) at each timestep
        potassium_conductance (np.ndarray): potassium conductance (v) at each timestep

    Returns:
        f_uv (np.ndarray): modified FHN term for membrane potential time derivative
        g_uv (np.ndarray): modified FHN term for potassium conductance time derivative
    """
    # TODO: CHECK ARRAY SIZES
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
    """Do one step of the temporal integration according to FHN term

    Args:
        new_membrane_potential (np.ndarray): container for the membrane potential at the n+1 
            time-step
        old_membrane_potential (np.ndarray): membrane potential at n time-step
        new_potassium_conductance (np.ndarray): container for the potassium conductance at the n+1 
            timestep
        old_potassium_conductance (np.ndarray): potassium conductance at n time-step
        temporal_step_size (float): time step
        f_uv (np.ndarray): FHN term for the membrane potential derivative at n time-step
        g_uv (np.ndarray): FHN term for the potassium conductance at n time-step
        diffusion_term (np.ndarray): diffusion term (depending on the spatial derivative)
        noise (np.ndarray): noise at n time-step. Should have (size-2) than the membrane potential 
            size
    """
    # TODO: CHECK ARRAY SIZES

    new_membrane_potential[1:-1, 1:-1] = (
        f_uv + diffusion_term + noise
    ) * temporal_step_size + old_membrane_potential[1:-1, 1:-1]
    new_potassium_conductance[1:-1, 1:-1] = (
        g_uv
    ) * temporal_step_size + old_potassium_conductance[1:-1, 1:-1]


def neumann_boundary_condition(
    membrane_potential: np.ndarray, potassium_conductance: np.ndarray
):
    """Calculate the boundary condition according the Neumann description (spatial derivative = 0)

    Args:
        membrane_potential (np.ndarray): membrane potential at timestep n+1
        potassium_conductance (np.ndarray): potassium conductance at timestep n+1
    """
    # TODO: CHECK ARRAY SIZES
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
    """Run the simulation with anisotropic diffusion of the media.

    Args:
        fitzhugh_nagumo_constants (FitzHughNagumoConstants): the constants for FHN
        discretisation_parameters (DiscretisationParameters): parameters for discretisation of the 
            simulation domain
        timepoints (np.ndarray): timepoints for the simulation
        excitable_media (ExcitableMedia): container for the values of the media
        noise (np.ndarray): noise that is generated
        diffusion_tensor (DiffusionTensor): diffusion tensor for each pixel of the simulation domain

    Returns:
        ExcitableMedia: values of the media after simulation
    """
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
    """Run the simulation with isotropic diffusion of the media.

    Args:
        fitzhugh_nagumo_constants (FitzHughNagumoConstants): the constants for FHN
        discretisation_parameters (DiscretisationParameters): parameters for discretisation of the 
            simulation domain
        timepoints (np.ndarray): timepoints for the simulation
        excitable_media (ExcitableMedia): container for the values of the media
        noise (np.ndarray): noise that is generated
        diffusion_constant (float): diffusion constant that is the same for each pixel in the media

    Returns:
        ExcitableMedia: values of the media after simulation
    """
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
