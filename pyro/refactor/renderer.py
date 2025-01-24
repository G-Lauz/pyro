import abc

import numpy

from pyro.refactor.geometry import Geometry
from pyro.refactor.kinematic import Kinematic


class Renderer(abc.ABC):
    @abc.abstractmethod
    def render(self, geometry: Geometry, kinematic: Kinematic, input_force: numpy.ndarray, trajectory: numpy.ndarray = None):
        pass
