import abc

from refactor.geometry import Geometry
from refactor.kinematic import Kinematic


class Renderer(abc.ABC):
    @abc.abstractmethod
    def render(self, geometry: Geometry, kinematic: Kinematic):
    # def render(self, geometry: Geometry, positions: numpy.ndarray):
        pass
