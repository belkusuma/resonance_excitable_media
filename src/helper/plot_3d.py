"""Convenience functions to make a 3D surface plot."""

import numpy as np
import matplotlib.pyplot as plt


def plot_3d_surface(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    x_label: str,
    y_label: str,
    z_label: str,
    linewidth: float = 0.1,
):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    X, Y = np.meshgrid(x_values, y_values)
    surf = ax.plot_surface(X, Y, z_values, linewidth=linewidth)  # type: ignore
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label) # type: ignore

    return fig
