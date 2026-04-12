"""
Metrics for analysis of spatiotemporal coherence.
Papers referenced in this file are:
{
M. Perc, ‘Spatial coherence resonance in excitable media’,
    Phys. Rev. E, vol. 72, no. 1, p. 016207, Jul. 2005,
    doi: 10.1103/PhysRevE.72.016207.
}
{
H. Busch and F. Kaiser, ‘Influence of spatiotemporally correlated noise on
    structure formation in excitable media’,
    Phys. Rev. E, vol. 67, no. 4, p. 041105, Apr. 2003,
    doi: 10.1103/PhysRevE.67.041105.
}
{
P. Jung, J. Wang, R. Wackerbauer, and K. Showalter,
    ‘Coherent structure analysis of spatiotemporal chaos’,
    Phys. Rev. E, vol. 61, no. 2, pp. 2095–2098, Feb. 2000,
    doi: 10.1103/PhysRevE.61.2095.
}


"""

import numpy as np
import scipy.fft
import scipy.interpolate

import skimage.measure

from src.helper.datatypes import DiscretisationParameters


def calculate_structure_function(
    values: np.ndarray,
    discretisation_parameters: DiscretisationParameters,
    start_time_index: int,
    end_time_index: int,
) -> np.ndarray:
    """
    Calculate the structure function as part of the analysis of spatiotemporal coherence.
    This would give a quantitative value of the spatial scale of the patterns emerging from
    stochastic resonance
    Taken from (Perc, 2005)


    Args:
        values (np.ndarray): 3d array of a 2D timeseries
        discretisation_parameters (DiscretisationParameters): the discretisation parameters for this
        start_time_index(int): the index of the timeseries to start the analysis
        end_time_index (int): the index of the timeseries to stop the analysis

    Returns:
        np.ndarray: a 2D array of the structure function, with the values for 0 frequency in the
            middle of the array
    """
    # Do a 2D FFT for the timepoints given.
    # The original paper did not disclose which timepoint they did this analysis on in each of the
    # different noise realisations, so we do several timepoints at the 'stable' region of the
    # time series
    fourier_transform = scipy.fft.fft2(
        values[:, :, start_time_index:end_time_index],
        axes=(0, 1),
    )
    # Shift so that the zero-frequency component is in the centre of the array, so that when plotted
    # and used for the path integrals, we don't have to shift again.
    fourier_shifted = scipy.fft.fftshift(fourier_transform, axes=(0, 1))

    # Take the mean over the timepoints given
    mean_power_spectra = np.mean(np.abs(fourier_shifted) ** 2, axis=2)

    # Return the power spectra / area of the grid
    return mean_power_spectra / (
        discretisation_parameters.grid_size[0]
        * discretisation_parameters.spatial_step_size[0]
        * discretisation_parameters.grid_size[1]
        * discretisation_parameters.spatial_step_size[1]
    )


def calculate_circular_line_integral(
    values: np.ndarray,
    grid_size: list[int],
    grid_step_size: list[float],
    radius: float,
    angular_step_size: float,
) -> float:
    """
    Numerically calculate a circular line integral (around the centre of the grid) over a grid of
    the values given.

    Args:
        values (np.ndarray): the values to integrate over. Should be a 2D array
        grid_size (tuple[int, int]): the grid size over which the values are distributed
        grid_step_size (tuple[float, float]): the step size of the grid over which the values are
            distributed
        radius (float): the radius of the circle to integrate over
        angular_step_size (float): the step size of the integration

    Returns:
        float: the value of the integral
    """
    # Calculate the number of steps to cover over the provided step size
    number_of_steps = int(2 * np.pi / angular_step_size)

    # Generate the grid given by the specification
    points = (
        np.arange(0, grid_size[0], grid_step_size[0]),
        np.arange(0, grid_size[0], grid_step_size[1]),
    )

    integrand = 0
    for i in range(number_of_steps):
        # Find the (x*, y*) coordinates of the point in the current integration step. Note that this
        # is from the centre of the grid
        x_star = (grid_size[0] / 2) + radius * np.cos(i * angular_step_size)
        y_star = (grid_size[1] / 2) + radius * np.sin(i * angular_step_size)

        # Interpolate the value at this coordinate from the given grid of values
        interpolated_value = scipy.interpolate.interpn(
            points, values, [x_star, y_star]
        )[0]
        integrand += interpolated_value * (radius * angular_step_size)

    return integrand

# TODO: make a function to find the peak for the p_k

def calculate_linear_cross_correlation(
    values: np.ndarray, discretisation_parameters: DiscretisationParameters
) -> float:
    """
    Calculate the linear cross correlation as a spatiotemporal measure of coherence, as described
    in (Busch and Kaiser, 2003)


    Args:
        values (np.ndarray): a 3D array of a 2D value timeseries
        discretisation_parameters (DiscretisationParameters): the discretisation parameters for this

    Returns:
        float: a value for the linear cross correlation
    """

    cross_correlation = np.zeros(values.shape[2] - 1)
    for t in range(values.shape[2]):
        # Calculate the mean value of a single timepoint
        mean_value = np.mean(values[:, :, t])

        # Calculate the variance for the timepoint as described in the paper
        variance = np.sum((values[:, :, t] - mean_value) ** 2) / (
            discretisation_parameters.grid_size[0]
            * discretisation_parameters.grid_size[1]
        )

        # Calculate the covariance for the timepoint as described in the paper
        covariance = calculate_covariance(values[:, :, t], float(mean_value))

        # If the variance is 0, this would lead to NaN because of the division, so exclue it
        if variance > 0:
            cross_correlation[t] = covariance / variance

    # The value for the timeseries is the mean of the linear cross-correlation over the timeseries
    return float(np.mean(cross_correlation))


def calculate_covariance(two_dimensional_values: np.ndarray, mean_value: float) -> float:
    """Calculate covariance according to the (Busch and Kaiser, 2003)

    Args:
        two_dimensional_values (np.ndarray): the 2D array to calculate the covariance for
        mean_value (float): the mean value of the 2D array

    Returns:
        float: covariance value
    """
    # Only for a square grid for now
    assert two_dimensional_values.shape[0] == two_dimensional_values.shape[1]
    grid_size = two_dimensional_values.shape[0]

    # Start with calculating the centre. In this, there's 4 cells for the von neumann neighbourhood
    von_neuman_neighbourhood_centre = np.empty((4, grid_size - 2, grid_size - 2))
    von_neuman_neighbourhood_centre[0, :, :] = two_dimensional_values[:-2, 1:-1]
    von_neuman_neighbourhood_centre[1, :, :] = two_dimensional_values[2:, 1:-1]
    von_neuman_neighbourhood_centre[2, :, :] = two_dimensional_values[1:-1, :-2]
    von_neuman_neighbourhood_centre[3, :, :] = two_dimensional_values[1:-1, 2:]

    covariance_centre = (
        np.sum(
            np.tile((two_dimensional_values[1:-1, 1:-1] - mean_value), reps=(4, 1, 1))
            * (von_neuman_neighbourhood_centre - mean_value)
        )
        / 4
    )

    # Calculate for the edges (x =0, grid size; y = 0, grid size). For this the cell in the von 
    # neumann neighbourhood is 3
    von_neuman_neighbourhood_edge = np.zeros((3, grid_size - 2, 4))
    values_edge = np.zeros((grid_size - 2, 4))

    # Left edge
    von_neuman_neighbourhood_edge[0, :, 0] = two_dimensional_values[0, :-2]
    von_neuman_neighbourhood_edge[1, :, 0] = two_dimensional_values[1, 1:-1]
    von_neuman_neighbourhood_edge[2, :, 0] = two_dimensional_values[0, 2:]
    values_edge[:, 0] = two_dimensional_values[0, 1:-1]

    # Right edge
    von_neuman_neighbourhood_edge[0, :, 1] = two_dimensional_values[-1, :-2]
    von_neuman_neighbourhood_edge[1, :, 1] = two_dimensional_values[-2, 1:-1]
    von_neuman_neighbourhood_edge[2, :, 1] = two_dimensional_values[-1, 2:]
    values_edge[:, 1] = two_dimensional_values[-1, 1:-1]

    # Top edge
    von_neuman_neighbourhood_edge[0, :, 2] = two_dimensional_values[:-2, 0]
    von_neuman_neighbourhood_edge[1, :, 2] = two_dimensional_values[1:-1, 1]
    von_neuman_neighbourhood_edge[2, :, 2] = two_dimensional_values[2:, 0]
    values_edge[:, 2] = two_dimensional_values[1:-1, 0]

    # Bottom edge
    von_neuman_neighbourhood_edge[0, :, 3] = two_dimensional_values[:-2, -1]
    von_neuman_neighbourhood_edge[1, :, 3] = two_dimensional_values[1:-1, -2]
    von_neuman_neighbourhood_edge[2, :, 3] = two_dimensional_values[2:, -1]
    values_edge[:, 3] = two_dimensional_values[1:-1, -1]

    covariance_edge = (
        np.sum(
            np.tile((values_edge - mean_value), reps=(3, 1, 1))
            * (von_neuman_neighbourhood_edge - mean_value)
        )
        / 3
    )


    # For the 4 corners of the grid
    von_neuman_neighbourhood_0 = np.array(
        [two_dimensional_values[1, 0], two_dimensional_values[0, 1]]
    )
    corner_term_0 = (
        np.sum(
            (two_dimensional_values[0, 0] - mean_value)
            * (von_neuman_neighbourhood_0 - mean_value)
        )
        / 2
    )

    von_neuman_neighbourhood_1 = np.array(
        [two_dimensional_values[-2, 0], two_dimensional_values[-1, 1]]
    )
    corner_term_1 = (
        np.sum(
            (two_dimensional_values[-1, 0] - mean_value)
            * (von_neuman_neighbourhood_1 - mean_value)
        )
        / 2
    )

    von_neuman_neighbourhood_2 = np.array(
        [two_dimensional_values[0, -2], two_dimensional_values[1, -1]]
    )
    corner_term_2 = (
        np.sum(
            (two_dimensional_values[0, -1] - mean_value)
            * (von_neuman_neighbourhood_2 - mean_value)
        )
        / 2
    )

    von_neuman_neighbourhood_3 = np.array(
        [two_dimensional_values[-1, -2], two_dimensional_values[-2, -1]]
    )
    corner_term_3 = (
        np.sum(
            (two_dimensional_values[-1, -1] - mean_value)
            * (von_neuman_neighbourhood_3 - mean_value)
        )
        / 2
    )

    return (
        covariance_centre
        + covariance_edge
        + corner_term_0
        + corner_term_1
        + corner_term_2
        + corner_term_3
    ) / (two_dimensional_values.shape[0] * two_dimensional_values.shape[1])



def calculate_spatiotemporal_entropy(
    three_dimensional: np.ndarray, threshold_value: float
) -> float:
    """Spatiotemporal entropy as a coherence metric. From Jung et al., 2000


    Args:
        three_dimensional (np.ndarray): a 3D array of a 2D timeseries
        threshold_value (float): a value for thresholding of 'active' clusters

    Returns:
        float: the spatiotemporal entropy
    """
    
    # Make a binary image of the 3D array, and then find the clusters that are connected to each
    # other across time and space
    binary = np.where(three_dimensional > threshold_value, 1, 0)
    labels, nums = skimage.measure.label(binary, connectivity=3, return_num=True) # type: ignore

    # Find the size of each of the clusters
    size_of_clusters = np.empty(nums - 1)
    for num in range(1, nums):
        size_of_cluster = np.sum(np.where(labels == num))
        size_of_clusters[num - 1] = size_of_cluster

    # Make a histogram of the size of the clusters (because none of the clusters will have the same
    # size), and then use the centre of the bin to be the size of the cluester
    hist, bin_edges = np.histogram(size_of_clusters, bins=20)
    sizes = (bin_edges[:-1] + bin_edges[1:]) / 2
    v_s = sizes * hist / (np.sum(sizes * hist))

    # Only use the non-zero v_s, because log(0) is undefined
    non_zero_vs = v_s[np.nonzero(v_s)]
    spatiotemporal_entropy = -1 * np.sum(non_zero_vs * np.log(non_zero_vs))

    return spatiotemporal_entropy


