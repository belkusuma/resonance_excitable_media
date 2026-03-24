"""Generate a difussion tensor from an image mask given."""
import numpy as np
import skimage.io as image_io
import skimage.color as color
import pathlib
from src.helper.datatypes import DiffusionTensor


def import_mask_of_tract(path : pathlib.Path):
    """Returns a binary mask where the pixel with the tract is true and the background is false"""
    img = image_io.imread(path)
    gray_img = color.rgb2gray(color.rgba2rgb(img))

    return np.invert(np.asarray(gray_img, dtype=bool))

def create_diffusion_tensor_from_mask(
    d_isotropic: float,
    d_anisotropic: list[tuple[float, float, float]],
    mask: list[np.ndarray]
) -> DiffusionTensor:
    """Create a (symmetric) diffusion tensor from the given masks and diffusion values.

    Args:
        d_isotropic (float): the diffusion value for everything outside the anisotropic area
        d_anisotropic (list[tuple[float, float, float]]): a list of anisotropic diffusion values.
            Each member of the list should be [d_xx, d_xy, d_yy]
        mask (list[np.ndarray]): a list of the anisotropic area masks (tract is true and background
            is false)

    Returns:
        DiffusionTensor: A tensor with the anisotropic diffusion values
    """
    # Combine all the masks into a single mask to get where the pixel is just a background
    background_mask = np.ones(mask[0].shape)
    for i in range(len(mask)):
        background_mask = background_mask - mask[i]

    
    # For the background, put in the isotropic value. Note that d_xy is 0 because isotropic
    # diffusion will have negligible value in the diagonal diffusion
    d_xx = d_isotropic * background_mask
    d_xy = np.zeros(mask[0].shape)
    d_yy = d_isotropic * background_mask
    
    # For each mask, add the values for the anisotropic diffusion to the whole tensor.
    for i in range(len(mask)):
        d_xx = (
            d_xx
            + d_anisotropic[i][0] * mask[i]
        )
        d_xy = d_xy + d_anisotropic[i][1] * mask[i]
        d_yy = (
            d_yy +
            + d_anisotropic[i][2] * mask[i]
        )
        
    return DiffusionTensor(d_xx, d_xy, d_yy)
