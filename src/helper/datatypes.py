"""File for all the datatypes."""
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import pathlib

import numpy as np
class NoiseType(Enum):
    """Noise type for generation."""
    WHITE = 0
    CORRELATED = 1
    NOT_IMPLEMENTED = 2

    @staticmethod
    def from_string(label):
        if label == "WHITE":
            return NoiseType.WHITE
        elif label == "CORRELATED":
            return NoiseType.CORRELATED
        else:
            return NoiseType.NOT_IMPLEMENTED

class DiffusionType(Enum):
    """Diffusion type."""
    ISOTROPIC = 0
    ANISOTROPIC = 1
    NOT_IMPLEMENTED = 2

    @staticmethod
    def from_string(label):
        if label == "ISOTROPIC":
            return DiffusionType.ISOTROPIC
        elif label == "ANISOTROPIC":
            return DiffusionType.ANISOTROPIC
        else:
            return DiffusionType.NOT_IMPLEMENTED

class SimulationType(Enum):
    NOISE_ONLY = 0
    FULL_SIMULATION = 1
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
class ExcitableMedia:
    """Description of an excitable media. Only FHN for now."""
    grid_size : tuple[int, int]
    temporal_size: int
    membrane_potential: np.ndarray = field(init=False)
    potassium_conductance: np.ndarray = field(init=False)

    def __post_init__(self):
        self.membrane_potential = np.zeros((self.grid_size[0], self.grid_size[1], self.temporal_size))
        self.potassium_conductance = np.zeros((self.grid_size[0], self.grid_size[1], self.temporal_size))

@dataclass
class DiffusionTensor:
    """(Symmetric) Tensor for diffusion constants."""
    d_xx: np.ndarray
    d_xy: np.ndarray
    d_yy: np.ndarray
    size :tuple[int, int] = field(init=False)

    def __post_init__(self):
        if not (np.all(self.d_xx.shape == self.d_xy.shape) and (np.all(self.d_xy.shape == self.d_yy.shape)) and (np.all(self.d_xx.shape ==self.d_yy.shape))):
            raise AttributeError
        
        self.size = self.d_xx.shape

@dataclass
class SimulationParameters:
    """Parameters for simulation that is then read from a json file."""
    fitzhugh_nagumo_constants: FitzHughNagumoConstants
    discretisation_parameters : DiscretisationParameters
    diffusion_type : DiffusionType
    diffusion_constant: float | DiffusionTensor
    simulation_time: float
    noise_type : NoiseType
    noise_intensity: float
    spatial_noise_correlation: float
    temporal_noise_correlation: float

    def __post_init__(self):
        if self.diffusion_type == DiffusionType.ISOTROPIC:
            diffusion_valid = isinstance(self.diffusion_constant, float)
        elif self.diffusion_type == DiffusionType.ANISOTROPIC:
            diffusion_valid = isinstance(self.diffusion_constant, DiffusionTensor)
        else:
            diffusion_valid = False
        
        if self.noise_type == NoiseType.WHITE:
            noise_valid = (np.isclose(self.spatial_noise_correlation, 0 ) and np.isclose(self.temporal_noise_correlation, 0))
        elif self.noise_type == NoiseType.CORRELATED:
            noise_valid = not (np.isclose(self.spatial_noise_correlation, 0 ) and np.isclose(self.temporal_noise_correlation, 0))
        else:
            noise_valid = False
        
        if not (diffusion_valid and noise_valid):
            raise AttributeError


@dataclass
class ParametersBatch:
    fitzhugh_nagumo_constants: FitzHughNagumoConstants
    discretisation_parameters : DiscretisationParameters
    simulation_time: float
    ensemble_number: int
    diffusion_type_array : list[str]
    diffusion_constant_array: list[float]
    noise_type_array : list[str]
    noise_intensity_array: list[float]
    spatial_correlation_array: list[float]
    temporal_correlation_array: list[float]
@dataclass
class EllipseParameters:
    x0 : float
    y0: float
    major_axis: float
    minor_axis: float
    angle_of_rotation: float
    eccentricity: float = field(init=False)

    def __post_init__(self):
        r = (self.minor_axis / self.major_axis) **2
        if r > 1:
            r = 1/r

        self.eccentricity = np.sqrt(1-r)
    def normalise(self):
        self.minor_axis = self.minor_axis/self.major_axis
        self.major_axis = 1
    
    def normalised(self) -> bool:
        if np.isclose(self.major_axis, 1.0):
            return True
        else:
            return False
        
    def to_json(self, path_to_save : pathlib.Path):
        data_dict = asdict(self)
        with open(path_to_save, 'w') as f:
            f.write(json.dumps(data_dict, indent=4))
    
    @staticmethod
    def from_json(path_to_file: pathlib.Path):
        with open(path_to_file) as f:
            data = json.load(f)

            return EllipseParameters(x0=data["x0"],
                                     y0=data["y0"],
                                     major_axis=data["major_axis"],
                                     minor_axis=data["minor_axis"],
                                     angle_of_rotation=data["angle_of_rotation"])
