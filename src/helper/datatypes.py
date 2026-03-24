from dataclasses import dataclass

import numpy as np

@dataclass
class FitzHughNagumoConstants:
    """Constants for FitzHugh-Nagumo equations."""
    a : float
    b: float
    epsilon: float

@dataclass
class DiscretisationParameters:
    """Discretisation parameters of the media."""
    grid_size: list[int] # Should be x, y
    spatial_step_size: list[float] # Should be dx, dy
    temporal_step_size: float

@dataclass
class SimulationParameters:
    """Parameters for simulation that is then read from a json file."""
    fitzhugh_nagumo_constants: FitzHughNagumoConstants
    discretisation_parameters : DiscretisationParameters
    diffusion_constant: float
    noise_intensity: float
    simulation_time: float

@dataclass
class ExcitableMedia:
    """Description of an excitable media. Only FHN for now."""
    membrane_potential: np.ndarray
    potassium_conductance: np.ndarray

@dataclass
class DiffusionTensor:
    """(Symmetric) Tensor for diffusion constants."""
    d_xx: np.ndarray
    d_xy: np.ndarray
    d_yy: np.ndarray