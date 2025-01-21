import abc

import numpy

from refactor.geometry import Geometry
from refactor.kinematic import Kinematic


class Renderer(abc.ABC):
    @abc.abstractmethod
    def render(self, geometry: Geometry, kinematic: Kinematic, input_force: numpy.ndarray, trajectory: numpy.ndarray = None):
        pass
