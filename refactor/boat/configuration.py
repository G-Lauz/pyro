import pathlib
from typing import List

import dataclasses

from refactor.configuration import Configuration


@dataclasses.dataclass
class QuadraticDamping:
    cx_max: float
    cy_max: float
    cm_max: float


@dataclasses.dataclass
class Parameters:
    dof: int

    state_upper_bound: List[float]
    state_lower_bound: List[float]
    
    input_upper_bound: List[float]
    input_lower_bound: List[float]

    force_inputs: int
    mass: float
    inertia: float

    linear_damping_coefficients: List[float]
    quadratic_damping_coefficients: QuadraticDamping

    water_density: float


@dataclasses.dataclass
class Kinematic:
    x: float
    y: float
    theta: float
    dx: float
    dy: float
    dtheta: float


@dataclasses.dataclass
class Geometry:
    thrust_offset: float
    lateral_area: float
    frontal_area: float
    length_overall: float


@dataclasses.dataclass
class Boat2DConfiguration(Configuration):
    name: str
    parameters: Parameters
    kinematic: Kinematic
    geometry: Geometry

    def __init__(self, config_file: pathlib.Path = None):
        super().__init__(config_file=config_file)
