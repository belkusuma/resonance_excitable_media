"""
Generating spatiotemporally correlated noise.
Adapted from
{
    J. García Ojalvo, J. M. Sancho, and L. Ramírez Piscina,
    ‘Generation of spatiotemporal colored noise’,
    1992, Accessed: Mar. 13, 2026. [Online].
    Available: https://hdl.handle.net/2445/9550
}
"""

import numpy as np
from scipy.fft import ifft2


def make_alpha_mu_upsilon(rng: np.random.Generator, grid_size: int) -> np.ndarray:
    """Generate the stochastic numbers with Gaussian distribution in the (spatial) frequency space.
    Is alpha_mu_upsilon in eq (32) of the paper, having correlation such as given in eq (33) and
    the generation of described in the Appendix of the paper

    Args:
        rng (np.random.Generator): random number generator
        grid_size (int): grid size (same in frequency or real space)

    Returns:
        np.ndarray: an array of random numbers with the given correlation and size
    """
    # See (A5) in the paper
    # Creating the coefficients with only grid_size/2 is due to the symmetry of the Fourier space
    # (see Appendix)
    a_mu_upsilon = (1 / np.sqrt(2)) * rng.standard_normal(
        size=(int(grid_size), int(grid_size / 2))
    )
    b_mu_upsilon = (1 / np.sqrt(2)) * rng.standard_normal(
        size=(int(grid_size), int(grid_size / 2))
    )

    # Piece the random coeeficients together. See Figure 4 of the paper
    alpha_mu_upsilon = np.empty((grid_size, grid_size), dtype=np.complex128)
    alpha_mu_upsilon[0:grid_size, 0 : int(grid_size / 2)] = (
        a_mu_upsilon + 1j * b_mu_upsilon
    )
    alpha_mu_upsilon[0:grid_size, int(grid_size / 2) :] = (
        a_mu_upsilon - 1j * b_mu_upsilon
    )

    # For the points of symmetricity, make sure that the values are only real (see Figure 4, (A6))
    for i in [0, int(grid_size / 2) - 1, grid_size - 1]:
        for j in [0, int(grid_size / 2) - 1, grid_size - 1]:
            alpha_mu_upsilon[i, j] = np.real(alpha_mu_upsilon[i, j])

    return alpha_mu_upsilon


def generate_spatiotemporal_correlated_noise(
    spatial_correlation: float,
    temporal_correlation: float,
    noise_intensity: float,
    spatial_step_size: float,
    temporal_step_size: float,
    grid_size: int,
    number_of_time_steps: int,
) -> np.ndarray:
    """Generate a spatiotemporally correlated noise. Immediately taken from the paper referenced
    above

    Args:
        spatial_correlation (float): The desired spatial correlation
        temporal_correlation (float): the desired temporal correlation
        noise_intensity (float): the desired noise intensity
        spatial_step_size (float): step size in the grid (spatial)
        temporal_step_size (float): time step size
        grid_size (int): the desired noise intensity (in real space)
        number_of_time_steps (int): number of time steps until we reach the desired time in
            simulation

    Returns:
        np.ndarray: noise that is correlated in time. Have size
            (grid_size, grid_size, number_of_time_steps)
    """
    # Condition of the noise generation. See eq (11)
    assert temporal_step_size < (
        temporal_correlation * (spatial_step_size**2) / (4 * (spatial_correlation**2))
    )
    rng = np.random.default_rng()

    mu = np.arange(0, grid_size, spatial_step_size)
    upsilon = np.arange(0, grid_size, spatial_step_size)

    # Eq (17) in the paper
    c_mu_upsilon = 1 - (2 * spatial_correlation**2 / (spatial_step_size**2)) * (
        np.cos(2 * np.pi * mu / grid_size) + np.cos(2 * np.pi * upsilon / grid_size) - 2
    )

    noise_mu_upsilon = np.empty(
        (grid_size, grid_size, number_of_time_steps), dtype=np.complex128
    )
    # First noise is given by this initial condition (eq (36))
    noise_mu_upsilon[:, :, 0] = np.sqrt(
        noise_intensity
        * (grid_size * spatial_step_size) ** 2
        / (temporal_correlation * c_mu_upsilon)
    ) * make_alpha_mu_upsilon(rng, grid_size)

    # A term that will be used in eq (34) that is constant in time, calculated here so that we don't
    # have to calculate it multiple times
    temp = np.sqrt(
        (
            noise_intensity
            * (grid_size * spatial_step_size) ** 2
            / (temporal_correlation * c_mu_upsilon)
        )
        * (1 - np.exp(-2 * c_mu_upsilon * temporal_step_size / temporal_correlation))
    )

    for t in range(1, number_of_time_steps):
        # Eq (34) in the paper
        noise_mu_upsilon[:, :, t] = noise_mu_upsilon[:, :, t - 1] * np.exp(
            -c_mu_upsilon * temporal_step_size / temporal_correlation
        ) + temp * make_alpha_mu_upsilon(rng, grid_size)

    # Move back to real space from the (spatial) frequency space by inverse FFT
    noise_real_space = np.zeros((grid_size, grid_size, number_of_time_steps))
    for t in range(number_of_time_steps - 1):
        noise_real_space[:, :, t] = np.real(ifft2(noise_mu_upsilon[:, :, t]))  # type: ignore

    return noise_real_space


def generate_white_noise(
    noise_intensity: float, grid_size: int, number_of_time_steps: int
) -> np.ndarray:
    """Generate a white (Gaussian) noise

    Args:
        noise_intensity (float): the desired noise intensity
        grid_size (int): the desired noise intensity
        number_of_time_steps (int): number of time steps until we reach the desired time in
            simulation

    Returns:
        np.ndarray: white noise with size (grid_size, grid_size, number_of_time_steps)
    """
    rng = np.random.default_rng()

    return noise_intensity * rng.standard_normal(
        size=(grid_size, grid_size, number_of_time_steps)
    )
