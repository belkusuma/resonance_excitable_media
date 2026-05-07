"""
{
R. Halır and J. Flusser,
    ‘NUMERICALLY STABLE DIRECT LEAST SQUARES FITTING OF ELLIPSES’,
    in Proc. 6th International Conference in Central Europe on Computer Graphics and Visualization,
    Feb. 1998, pp. 125–132.
}
"""

import numpy as np
from skimage import measure

from src.prototype.prototype_datatypes import EllipseParameters


def fit_ellipse_to_power_spectra(power_spectra: np.ndarray, contour_scale: float = 50)-> EllipseParameters:
    power_spectra_norm = power_spectra / np.max(power_spectra)
    contours = measure.find_contours(
        power_spectra_norm, contour_scale * np.min(power_spectra_norm)
    )

    # Get the longest contour of the found contour lines to fit to an ellipse
    longest_contour = max(contours, key=len)

    # Have to flip because measure.find_contours return [row, column]
    ellipse_cartesian_coefficient = fit_ellipse(
        longest_contour[:, 1], longest_contour[:, 0]
    )
    if not (len(ellipse_cartesian_coefficient) == 6):
        return EllipseParameters(
            x0=64, y0=64, major_axis=1, minor_axis=1, angle_of_rotation=0
        )
    ellipse_parameters = ellipse_cartesian_to_polar(ellipse_cartesian_coefficient)
    ellipse_parameters.normalise()

    return ellipse_parameters


def fit_ellipse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit the coefficients a, b, c, d, e, f representing an ellipse described by the formula
    F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided arrays of datapoints
    x =[x1, ... xn] and y = [y1, ..., yn].

    Based on the algorithm of Halir and Flusser, 1998.
    Adapted from https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/

    Args:
        x (np.ndarray): 1D array of the x-values
        y (np.ndarray): 1D array of the y-values

    Returns:
        np.ndarray: an array of 6 elements of the coefficients.
    """
    D1 = np.column_stack((x**2, x * y, y**2))
    D2 = np.column_stack((x, y, np.ones(len(x))))

    S1 = np.matmul(np.transpose(D1), D1)
    S2 = np.matmul(np.transpose(D1), D2)
    S3 = np.matmul(np.transpose(D2), D2)

    C1 = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]], dtype=np.float32)

    T = -1 * (np.matmul(np.linalg.inv(S3), np.transpose(S2)))
    M = S1 + np.matmul(S2, T)
    M = np.matmul(np.linalg.inv(C1), M)

    _, eigvec = np.linalg.eig(M)

    # Condition for an ellipse is that 4ac-b^2 > 0
    cond = 4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2
    a1 = eigvec[:, np.nonzero(cond > 0)[0]]

    return np.concatenate((a1, np.matmul(T, a1))).ravel()


def ellipse_cartesian_to_polar(cartesian_coefficients: np.ndarray) -> EllipseParameters:
    """
    Convert the cartesian conic coefficients (a, b, c, d, f, g) to the ellipse parameters, where
    F(x, y) = ax^2 + bxy + cy^2 + dx + fy + g = 0. Returned the ellipse parameters.

    Adapted from https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/

    Args:
        cartesian_coefficients (np.ndarray): an array of 6 elements that are the coefficients

    Raises:
        ValueError: _description_

    Returns:
        EllipseParameters: the parameters of an ellipse
    """
    # Check that we have the right number of coefficients
    assert len(cartesian_coefficients) == 6

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = cartesian_coefficients[0]
    b = cartesian_coefficients[1] / 2
    c = cartesian_coefficients[2]
    d = cartesian_coefficients[3] / 2
    f = cartesian_coefficients[4] / 2
    g = cartesian_coefficients[5]

    # Check that these coefficients give an ellipse
    den = b**2 - a * c
    if den > 0:
        raise ValueError("These coefficients do not give an ellipse!")

    # The location of the ellipse centre
    x0, y0 = (c * d - b * f) / den, (a * f - b * d) / den

    num = 2 * (a * f**2 + c * d**2 + g * b**2 - 2 * b * d * f - a * c * g)
    fac = np.sqrt((a - c) ** 2 + 4 * b**2)

    # Semi-major and semi-minor axis lengtsh (unsorted)
    major_axis = np.sqrt(num / (den * (fac - a - c)))
    minor_axis = np.sqrt(num / (den * (-fac - a - c)))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if major_axis < minor_axis:
        width_gt_height = False
        major_axis, minor_axis = minor_axis, major_axis

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi / 2
    else:
        phi = np.arctan(2 * b / (a - c)) / 2
        if a > c:
            phi += np.pi / 2

    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi / 2

    if isinstance(phi, float):
        phi = phi % np.pi
    else:
        phi = np.float64(0)

    return EllipseParameters(
        x0=x0,
        y0=y0,
        major_axis=major_axis,
        minor_axis=minor_axis,
        angle_of_rotation=phi,
    )
