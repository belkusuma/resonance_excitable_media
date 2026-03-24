"""Generate a difussion tensor from an image mask given."""
import numpy as np
import skimage.io as image_io
import skimage.color as color
import pathlib
from datatypes import DiffusionTensor


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
    background_mask = np.ones(mask[0].shape)
    for i in range(len(mask)):
        background_mask = background_mask - mask[i]

    
    d_xx = d_isotropic * background_mask
    d_xy = np.zeros(mask[0].shape)
    d_yy = d_isotropic * background_mask
    
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