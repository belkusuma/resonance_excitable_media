"""
Generating spatiotemporally correlated noise
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


def make_alpha_mu_upsilon(rng: np.random.Generator, grid_size: int):
    a_mu_upsilon = (1 / np.sqrt(2)) * rng.standard_normal(
        size=(int(grid_size), int(grid_size / 2))
    )
    b_mu_upsilon = (1 / np.sqrt(2)) * rng.standard_normal(
        size=(int(grid_size), int(grid_size / 2))
    )

    alpha_mu_upsilon = np.empty((grid_size, grid_size), dtype=np.complex128)
    alpha_mu_upsilon[0:grid_size, 0 : int(grid_size / 2)] = (
        a_mu_upsilon + 1j * b_mu_upsilon
    )
    alpha_mu_upsilon[0:grid_size, int(grid_size / 2) :] = (
        a_mu_upsilon - 1j * b_mu_upsilon
    )

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
    assert temporal_step_size < (
        temporal_correlation * (spatial_step_size**2) / (4 * (spatial_correlation**2))
    )
    rng = np.random.default_rng()

    mu = np.arange(0, grid_size, spatial_step_size)
    upsilon = np.arange(0, grid_size, spatial_step_size)
    c_mu_upsilon = 1 - (2 * spatial_correlation**2 / (spatial_step_size**2)) * (
        np.cos(2 * np.pi * mu / grid_size) + np.cos(2 * np.pi * upsilon / grid_size) - 2
    )

    noise_mu_upsilon = np.empty(
        (grid_size, grid_size, number_of_time_steps), dtype=np.complex128
    )
    noise_mu_upsilon[:, :, 0] = np.sqrt(
        noise_intensity
        * (grid_size * spatial_step_size) ** 2
        / (temporal_correlation * c_mu_upsilon)
    ) * make_alpha_mu_upsilon(rng, grid_size)

    temp = np.sqrt(
        (
            noise_intensity
            * (grid_size * spatial_step_size) ** 2
            / (temporal_correlation * c_mu_upsilon)
        )
        * (1 - np.exp(-2 * c_mu_upsilon * temporal_step_size / temporal_correlation))
    )

    for t in range(1, number_of_time_steps):
        noise_mu_upsilon[:, :, t] = noise_mu_upsilon[:, :, t - 1] * np.exp(
            -c_mu_upsilon * temporal_step_size / temporal_correlation
        ) + temp * make_alpha_mu_upsilon(rng, grid_size)

    noise_real_space = np.zeros((grid_size, grid_size, number_of_time_steps))
    for t in range(number_of_time_steps - 1):
        noise_real_space[:, :, t] = np.real(ifft2(noise_mu_upsilon[:, :, t]))

    return noise_real_space


def generate_white_noise(
    noise_intensity: float, grid_size: int, number_of_time_steps: int
) -> np.ndarray:
    rng = np.random.default_rng()

    return noise_intensity * rng.standard_normal(
        size=(grid_size, grid_size, number_of_time_steps)
    )
