import dataclasses

from typing import List

from pyro.refactor.dynamics.mechanical.configuration import MechanicalConfiguration


@dataclasses.dataclass
class QuadraticDampingConfiguration:
    cx_max: float
    cy_max: float
    cm_max: float


@dataclasses.dataclass
class HydrodynamicsConfiguration:
    water_density: float
    linear_damping: List[float]
    quadratic_damping: QuadraticDampingConfiguration


@dataclasses.dataclass
class BoatGeometryConfiguration:
    thrust_offset: float
    lateral_area: float
    frontal_area: float
    length_overall: float


@dataclasses.dataclass
class BoatConfiguration:
    mechanical: MechanicalConfiguration
    hydrodynamics: HydrodynamicsConfiguration
    geometry: BoatGeometryConfiguration
