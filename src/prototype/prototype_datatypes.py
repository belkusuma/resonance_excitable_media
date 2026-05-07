from dataclasses import dataclass, field, asdict
import json
import pathlib
import numpy as np

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