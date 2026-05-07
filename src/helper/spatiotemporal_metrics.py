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
"""

import numpy as np
import scipy.fft

from src.helper.datatypes import (
    DiscretisationParameters,
    CrossCorrelationType
)

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

def calculate_linear_cross_correlation_directed(
    values: np.ndarray,
    discretisation_parameters: DiscretisationParameters,
    cross_correlation_type: CrossCorrelationType,
    direction_vector: np.ndarray,
    start_time_index: int,
    end_time_index: int,
)-> float:
    """
    Calculate the directed linear cross correlation as a spatiotemporal measure of coherence,
    inspired by Busch and Kaiser (2003).
    Note that this value is calculated over the grid elements that are NOT the edges.

    Args:
        values (np.ndarray): a 3D array of a 2D value timeseries
        discretisation_parameters (DiscretisationParameters): the discretisation parameters for this
        cross_correlation_type (CrossCorrelationType): the type of cross correlation calculation
        direction_vector (np.ndarray): a vector of the direction the cross-correlation is 
            calculated against. Will NOT be used for CrossCorrelationType.MEAN
        start_time_index (int): the index for starting the calculation as time of the 3D array
        end_time_index (int): the index for ending the calculation as time of the 3D array

    Returns:
        float: a value for the directed linear cross correlation
    """
    number_of_time = end_time_index - start_time_index
    cross_correlation = np.zeros(number_of_time)

    for k in range(number_of_time):
        # Get the actual index of the 3D array
        t = start_time_index + k
        
        # Use only the non-edge values to calculate this metric
        mean_value = np.mean(values[1:-1, 1:-1, t])

        # Calculate the variance for this timepoint
        variance = np.sum(
            (values[1:-1, 1:-1, t] - mean_value) ** 2
            / (
                (discretisation_parameters.grid_size[0] - 2) #-2 because removing the edges
                * (discretisation_parameters.grid_size[1] - 2)
            )
        )

        # Calculate the covariance for this timepoint
        covariance = calculate_covariance_directed(
            values[:, :, t], float(mean_value), cross_correlation_type, direction_vector
        )

        # If the variance is 0, this would lead to NaN because of the division, so exclude it
        if variance > 0:
            cross_correlation[k] = covariance / variance
    
    return float(np.mean(cross_correlation))

def calculate_covariance_directed(
    two_dimensional_values: np.ndarray,
    mean_value: float,
    cross_correlation_type: CrossCorrelationType,
    direction_vector: np.ndarray,
)-> float:
    """
    Calculate the directed covariance. Inspired by Busch and Kaiser (2003) and Moran's index. See
    poster for full detail.

    Args:
        two_dimensional_values (np.ndarray): the 2D array to calculate the covariance for
        mean_value (float): the mean value of the 2D array
        cross_correlation_type (CrossCorrelationType): the type of cross correlation calculation
        direction_vector (np.ndarray): a vector of the direction the cross-correlation is 
            calculated against. Will NOT be used for CrossCorrelationType.MEAN

    Returns:
        float: a value for the directed covariance
    """
    # Only for a square grid for now
    assert two_dimensional_values.shape[0] == two_dimensional_values.shape[1]
    grid_size = two_dimensional_values.shape[0]

    # For convenience of array based calculations later
    neighbourhood_values = np.empty((8, grid_size - 2, grid_size - 2))
    neighbourhood_values[0, :, :] = two_dimensional_values[:-2, 1:-1] # (-1, 0)
    neighbourhood_values[1, :, :] = two_dimensional_values[2:, 1:-1] # (1, 0)
    neighbourhood_values[2, :, :] = two_dimensional_values[1:-1, :-2] # (0, -1)
    neighbourhood_values[3, :, :] = two_dimensional_values[1:-1, 2:] # (0, 1)
    neighbourhood_values[4, :, :] = two_dimensional_values[:-2, :-2] # (-1, -1)
    neighbourhood_values[5, :, :] = two_dimensional_values[2:, :-2] # (1, -1)
    neighbourhood_values[6, :, :] = two_dimensional_values[:-2, 2:] # (-1, 1)
    neighbourhood_values[7, :, :] = two_dimensional_values[2:, 2:] # (1, 1)

    if cross_correlation_type == CrossCorrelationType.MEAN:
        # Calculate the 'regular' cross correlation (as written in Busch and Kaiser, 2003)
        covariance = np.sum(
            np.tile((two_dimensional_values[1:-1, 1:-1] - mean_value), reps=(8, 1, 1))
            * (neighbourhood_values - mean_value)
            / 8
        )

    elif cross_correlation_type == CrossCorrelationType.DIRECTED:
        # Normalise the fiber direction vector first
        direction_vector_norm = np.sqrt(
            direction_vector[0] ** 2 + direction_vector[1] ** 2
        )
        direction_vector = direction_vector / direction_vector_norm

        # Calculate what the coefficients are for each direction of the Moore neighbourhood and the
        # direction vector
        coefficients = np.empty(8)
        coefficients[0] = np.abs(np.dot([-1, 0], direction_vector))
        coefficients[1] = np.abs(np.dot([1, 0], direction_vector))
        coefficients[2] = np.abs(np.dot([0, -1], direction_vector))
        coefficients[3] = np.abs(np.dot([0, 1], direction_vector))
        coefficients[4] = np.abs(np.dot([-1, -1], direction_vector))
        coefficients[5] = np.abs(np.dot([1, -1], direction_vector))
        coefficients[6] = np.abs(np.dot([-1, 1], direction_vector))
        coefficients[7] = np.abs(np.dot([1, 1], direction_vector))

        # Calculate the difference in values with array operations (much more efficient)
        difference_in_values = np.tile(
            (two_dimensional_values[1:-1, 1:-1] - mean_value), reps=(8, 1, 1)
        ) * (neighbourhood_values - mean_value)

        # Multiply the difference in values with the previosly calculated coefficients
        for i in range(8):
            difference_in_values[i, :, :] = coefficients[i] * difference_in_values[i, :, :]

        covariance = np.sum(difference_in_values) / np.sum(coefficients)
    else:
        # Should not get here, but keeps typecheck happy
        covariance = 0

    return covariance / ((two_dimensional_values.shape[0] - 2) ** 2)
