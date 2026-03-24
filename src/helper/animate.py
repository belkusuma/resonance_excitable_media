"""Animate a 3D array into a 2D image in time."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pathlib


def animate_plot(
    values: np.ndarray, timepoints: np.ndarray, cmap: str = "jet"
) -> animation.ArtistAnimation:
    """Animate the 3D array into a 2D images in time.

    Args:
        values (np.ndarray): the 3D values to animate
        timepoints (np.ndarray): timepoints to animate it in
        cmap (str, optional): The colour map for the image. Defaults to "jet".

    Returns:
        animation.ArtistAnimation: animation object (to either save or show)
    """
    fig, ax = plt.subplots()
    ims = []
    for t in range(len(timepoints) - 1):
        im = ax.imshow(
            values[:, :, t],
            vmin=np.min(values),
            vmax=np.max(values),
            cmap=cmap,
        )
        ims.append([im])

    return animation.ArtistAnimation(fig, ims)


def save_animation(animation: animation.ArtistAnimation, animation_path: pathlib.Path):
    """Save the animation with the given path."""
    animation.save(animation_path)


