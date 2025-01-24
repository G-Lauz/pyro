import abc

import numpy

from pyro.refactor.system import MechanicalSystem


class Renderer(abc.ABC):
    @abc.abstractmethod
    def render(self, system: MechanicalSystem, input_force: numpy.ndarray, trajectory: numpy.ndarray = None):
        pass
