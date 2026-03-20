from dataclasses import dataclass

import numpy as np

@dataclass
class FitzHughNagumoConstants:
    a : float
    b: float
    epsilon: float

@dataclass
class DiscretisationParameters:
    grid_size: list[int] # Should be x, y
    spatial_step_size: list[float] # Should be dx, dy
    temporal_step_size: float

@dataclass
class ExcitableMedia:
    membrane_potential: np.ndarray
    potassium_conductance: np.ndarray

@dataclass
class DiffusionTensor:
    d_xx: np.ndarray
    d_xy: np.ndarray
    d_yy: np.ndarray