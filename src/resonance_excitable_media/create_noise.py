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
    # Calculate the middle index for convenience
    mid_index = int(grid_size / 2)

    # Many of the proceeding operations only work if the grid size is even, so adjust the grid size
    # to be even.
    # If the grid size is odd, then the "the bottom row and the right column are identical with the
    # top row and left column" (see Appendix A) and so we can safely adjust the grid size for now
    if grid_size % 2 == 0:
        adjusted_grid_size = grid_size
    else:
        adjusted_grid_size = grid_size - 1

    a_mu_upsilon = np.empty((grid_size, grid_size))
    b_mu_upsilon = np.empty((grid_size, grid_size))

    # The standard deviation for the grid is 1/sqrt(2), except for the black sites on Fig 4a
    # (See A5)
    standard_deviation = 1 / np.sqrt(2)

    # Generate the random numbers for a_mu_upsilon
    # For the 'black sites' of Figure 4a, the variance is 1 for a_mu_upsilon
    a_mu_upsilon[0, 0] = rng.standard_normal()
    a_mu_upsilon[mid_index, 0] = rng.standard_normal()
    a_mu_upsilon[0, mid_index] = rng.standard_normal()
    a_mu_upsilon[mid_index, mid_index] = rng.standard_normal()

    # For the half top edge of the a_mu_upsilon grid (without the black sites)
    a_mu_upsilon[1:mid_index, 0] = standard_deviation * rng.standard_normal(
        size=mid_index - 1
    )
    # For the half left edge of the grid (without the black sites)
    a_mu_upsilon[0, 1:mid_index] = standard_deviation * rng.standard_normal(
        size=mid_index - 1
    )
    # For the half middle column of the grid (without the black sites)
    a_mu_upsilon[mid_index, 1:mid_index] = standard_deviation * rng.standard_normal(
        size=mid_index - 1
    )
    # For the rest of the shadowed cells in Figure 4b
    a_mu_upsilon[1:mid_index, 1:adjusted_grid_size] = (
        standard_deviation
        * rng.standard_normal(size=(mid_index - 1, adjusted_grid_size - 1))
    )

    # For the non-shadowed cells in Figure 4b, utilise the symmetry relation to fill the rest of the
    # grid. (See A3 and Figure 4a)
    # Because a_mu_upsilon is the real part, we are just flipping the order of the numbers in the
    # grid based on the symmetry relation
    # With the standard quadrant convention, with the axis in the middle of the grid,
    # the first quadrant is the flip of the third quadrant (without the middle of the grid)
    a_mu_upsilon[
        mid_index + 1 : adjusted_grid_size, mid_index + 1 : adjusted_grid_size
    ] = np.flip(a_mu_upsilon[1:mid_index, 1:mid_index])
    # Fourth quadrant is the flip of the second quadrant
    a_mu_upsilon[mid_index + 1 : adjusted_grid_size, 1:mid_index] = np.flip(
        a_mu_upsilon[1:mid_index, mid_index + 1 : adjusted_grid_size]
    )
    # Now we flip the numbers in the middle of the grid
    a_mu_upsilon[mid_index, mid_index + 1 : adjusted_grid_size] = np.flip(
        a_mu_upsilon[mid_index, 1:mid_index]
    )
    a_mu_upsilon[mid_index + 1 : adjusted_grid_size, mid_index] = np.flip(
        a_mu_upsilon[1:mid_index, mid_index]
    )
    a_mu_upsilon[0, mid_index + 1 : adjusted_grid_size] = np.flip(
        a_mu_upsilon[0, 1:mid_index]
    )
    a_mu_upsilon[mid_index + 1 : adjusted_grid_size, 0] = np.flip(
        a_mu_upsilon[1:mid_index, 0]
    )

    # Similarly, generate the b_mu_upsilon (imaginary part)
    # Note that for the imaginary part, the 'black sites' are zero (see A6)
    # For the half top edge of the a_mu_upsilon grid (without the black sites)
    b_mu_upsilon[1:mid_index, 0] = standard_deviation * rng.standard_normal(
        size=mid_index - 1
    )
    # For the half left edge of the grid (without the black sites)
    b_mu_upsilon[0, 1:mid_index] = standard_deviation * rng.standard_normal(
        size=mid_index - 1
    )
    # For the half middle column of the grid (without the black sites)
    b_mu_upsilon[mid_index, 1:mid_index] = standard_deviation * rng.standard_normal(
        size=mid_index - 1
    )
    # For the rest of the shadowed cells in Figure 4b
    b_mu_upsilon[1:mid_index, 1:adjusted_grid_size] = (
        standard_deviation
        * rng.standard_normal(size=(mid_index - 1, adjusted_grid_size - 1))
    )

    # For the non-shadowed cells in Figure 4b, utilise the symmetry relation to fill the rest of the
    # grid. (See A3 and Figure 4a)
    # For the imaginary part, we have to flip the order of the number in the grid and also the sign
    # of the number
    # With the standard quadrant convention, with the axis in the middle of the grid,
    # the first quadrant is the flip of the third quadrant (without the middle of the grid)
    b_mu_upsilon[
        mid_index + 1 : adjusted_grid_size, mid_index + 1 : adjusted_grid_size
    ] = -1 * np.flip(b_mu_upsilon[1:mid_index, 1:mid_index])
    # Fourth quadrant is the flip of the second quadrant
    b_mu_upsilon[mid_index + 1 : adjusted_grid_size, 1:mid_index] = -1 * (
        np.flip(a_mu_upsilon[1:mid_index, mid_index + 1 : adjusted_grid_size])
    )
    # Now we flip the numbers in the middle of the grid
    b_mu_upsilon[mid_index, mid_index + 1 : adjusted_grid_size] = -1 * np.flip(
        b_mu_upsilon[mid_index, 1:mid_index]
    )
    b_mu_upsilon[mid_index + 1 : adjusted_grid_size, mid_index] = -1 * np.flip(
        b_mu_upsilon[1:mid_index, mid_index]
    )

    b_mu_upsilon[0, mid_index + 1 : adjusted_grid_size] = -1 * np.flip(
        b_mu_upsilon[0, 1:mid_index]
    )
    b_mu_upsilon[mid_index + 1 : adjusted_grid_size, 0] = -1 * np.flip(
        b_mu_upsilon[1:mid_index, 0]
    )

    # If the size of the grid is odd, make the right column equal to the left column, and the
    # bottom row equal to the top row
    if grid_size % 2 == 1:
        a_mu_upsilon[:, 0] = a_mu_upsilon[:, grid_size]
        a_mu_upsilon[0, :] = a_mu_upsilon[grid_size, :]
        b_mu_upsilon[:, 0] = b_mu_upsilon[:, grid_size]
        b_mu_upsilon[0, :] = b_mu_upsilon[grid_size, :]

    return a_mu_upsilon + 1j * b_mu_upsilon


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
    rng = np.random.default_rng()

    mu = np.linspace(0, spatial_step_size * grid_size, num=grid_size)
    upsilon = np.linspace(0, spatial_step_size * grid_size, num=grid_size)

    mu_meshgrid, upsilon_meshgrid = np.meshgrid(mu, upsilon)
    # Eq (17) in the paper
    c_mu_upsilon = 1 - (2 * spatial_correlation**2 / (spatial_step_size**2)) * (
        np.cos(2 * np.pi * mu_meshgrid / grid_size)
        + np.cos(2 * np.pi * upsilon_meshgrid / grid_size)
        - 2
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
