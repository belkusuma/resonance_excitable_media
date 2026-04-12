"""Convenience functions to make a 3D surface plot."""
import numpy as np
import matplotlib.pyplot as plt

def plot_3d_surface(x_values: np.ndarray, y_values: np.ndarray, z_values: np.ndarray, linewidth: float = 0.1):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    X, Y = np.meshgrid(x_values, y_values)
    surf = ax.plot_surface(X, Y, z_values, linewidth = linewidth) # type: ignore

    return fig