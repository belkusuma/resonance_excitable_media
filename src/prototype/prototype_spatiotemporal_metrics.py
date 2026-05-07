"""
Spatiotemporal metrics that were considered in the process of creating this script, but abandoned
for now. 

{
P. Jung, J. Wang, R. Wackerbauer, and K. Showalter,
    ‘Coherent structure analysis of spatiotemporal chaos’,
    Phys. Rev. E, vol. 61, no. 2, pp. 2095–2098, Feb. 2000,
    doi: 10.1103/PhysRevE.61.2095.
}
"""
import numpy as np
from scipy.interpolate import interpn

from skimage.measure import label
from src.prototype.prototype_datatypes import EllipseParameters

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
        np.arange(0, grid_size[1], grid_step_size[1]),
    )

    angle_array = np.linspace(0, 2 * np.pi, number_of_steps)
    integrand = 0
    for angle in angle_array:
        # Find the (x*, y*) coordinates of the point in the current integration step. Note that this
        # is from the centre of the grid
        x_star = (grid_size[0] / 2) + radius * np.cos(angle)
        y_star = (grid_size[1] / 2) + radius * np.sin(angle)

        # Interpolate the value at this coordinate from the given grid of values
        interpolated_value = interpn(
            points, values, [x_star, y_star]
        )[0]

        # Get the integrand and add it to the running total
        integrand += interpolated_value * (radius * angular_step_size)

    return integrand


def calculate_elliptic_line_integral(
    values: np.ndarray,
    grid_size: list[int],
    grid_step_size: list[float],
    ellipse_parameters: EllipseParameters,
    scale_factor: float,
    angular_step_size: float,
) -> float:
    """
    Numerically calculate a elliptical line integral over a grid of
    the values given.

    Args:
        values (np.ndarray): the values to integrate over. Should be a 2D array
        grid_size (tuple[int, int]): the grid size over which the values are distributed
        grid_step_size (tuple[float, float]): the step size of the grid over which the values are
            distributed
        ellipse_parameters (EllipseParameters): the parameters given for the ellipse. The instance
            will be normalised in the process
        scale_factor (float): the scaling of the semi-major and semi-minor axes
        angular_step_size (float): the step size of the integration

    Returns:
        float: the value of the integral
    """
    # Calculate the number of steps to cover over the provided angular step size
    number_of_steps = int(2 * np.pi / angular_step_size)

    # Generate the grid given by the specification
    points = (
        np.arange(0, grid_size[0], grid_step_size[0]),
        np.arange(0, grid_size[1], grid_step_size[1]),
    )

    # Normalise the ellipse parameters if not yet normalised
    if not ellipse_parameters.normalised:
        ellipse_parameters.normalise()

    t_array = np.linspace(0, 2 * np.pi, number_of_steps)
    integrand = 0
    for t in t_array:
        # Find the (x*, y*) coordinates of the point in the current integration step. Note that this
        # is from the centre of the grid
        x_star = (
            ellipse_parameters.x0
            + (
                scale_factor
                * ellipse_parameters.major_axis
                * np.cos(t)
                * np.cos(ellipse_parameters.angle_of_rotation)
            )
            - (
                scale_factor
                * ellipse_parameters.minor_axis
                * np.sin(t)
                * np.sin(ellipse_parameters.angle_of_rotation)
            )
        )
        y_star = (
            ellipse_parameters.y0
            + (
                scale_factor
                * ellipse_parameters.major_axis
                * np.cos(t)
                * np.sin(ellipse_parameters.angle_of_rotation)
            )
            + (
                scale_factor
                * ellipse_parameters.minor_axis
                * np.sin(t)
                * np.cos(ellipse_parameters.angle_of_rotation)
            )
        )

        # Calculate the derivative of x and y to calculate the integrand
        x_prime = (
            scale_factor
            * ellipse_parameters.major_axis
            * -np.sin(t)
            * np.cos(ellipse_parameters.angle_of_rotation)
        ) - (
            scale_factor
            * ellipse_parameters.minor_axis
            * np.cos(t)
            * np.sin(ellipse_parameters.angle_of_rotation)
        )
        y_prime = (
            scale_factor
            * ellipse_parameters.major_axis
            * -np.sin(t)
            * np.sin(ellipse_parameters.angle_of_rotation)
        ) + (
            scale_factor
            * ellipse_parameters.minor_axis
            * np.cos(t)
            * np.cos(ellipse_parameters.angle_of_rotation)
        )

        # Because of the angle of the ellipse and the ratio between the semi-major and semi-minor
        # axis, (x*, y*) coordinates might not actually end up inside the grid
        try:
            # Interpolate the value at this coordinate from the given grid of values
            interpolated_value = interpn(
                points, values, [x_star, y_star]
            )[0]

            # Calculate the integrand and add it to the running total
            integrand += (
                interpolated_value
                * np.sqrt(x_prime**2 + y_prime**2)
                * angular_step_size
            )

        except:
            # If one of the values is not within the grid, abort and return 0
            integrand = 0
            break

    return integrand


def calculate_ellipse_perimeter(
    ellipse_parameter: EllipseParameters, scale_factor: float = 1.0
) -> float:
    """Ramanujan's first approximation for a perimeter of an ellipse.

    Args:
        ellipse_parameter (EllipseParameters): the elliptical parameters
        scale_factor (float, optional): scale factor for the semi-major and semi-minor axes. Defaults to 1.0.

    Returns:
        float: the (approximate) perimeter of the ellipse.
    """
    return np.pi * (
        3 * scale_factor * (ellipse_parameter.major_axis + ellipse_parameter.minor_axis)
        - np.sqrt(
            scale_factor**2
            * (3 * ellipse_parameter.major_axis + ellipse_parameter.minor_axis)
            * (ellipse_parameter.major_axis + 3 * ellipse_parameter.minor_axis)
        )
    )

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

    # Make a binary image of the 3D array, and then find the clusters that are connected to each
    # other across time and space
    binary = np.where(three_dimensional > threshold_value, 1, 0)
    labels, nums = label(binary, connectivity=3, return_num=True)  # type: ignore

    # Find the size of each of the clusters
    size_of_clusters = np.empty(nums - 1)
    for num in range(1, nums):
        size_of_cluster = np.sum(np.where(labels == num))
        size_of_clusters[num - 1] = size_of_cluster

    # Make a histogram of the size of the clusters (because none of the clusters will have the same
    # size), and then use the centre of the bin to be the size of the cluester
    hist, bin_edges = np.histogram(size_of_clusters, bins=20)
    sizes = (bin_edges[:-1] + bin_edges[1:]) / 2
    v_s = sizes * hist / (np.sum(sizes * hist))

    # Only use the non-zero v_s, because log(0) is undefined
    non_zero_vs = v_s[np.nonzero(v_s)]
    spatiotemporal_entropy = -1 * np.sum(non_zero_vs * np.log(non_zero_vs))

    return spatiotemporal_entropy

