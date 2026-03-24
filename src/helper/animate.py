import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pathlib


def animate_plot(
    values: np.ndarray, timepoints: np.ndarray, cmap: str = "jet"
) -> animation.ArtistAnimation:
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
    animation.save(animation_path)


