import abc

import numpy

from pyro.refactor.signal import StateSignal, Signal
from pyro.refactor.dynamics.mechanical.geometry import Geometry


class Renderer(abc.ABC):
    def __init__(self, geometry: Geometry):
        self.geometry = geometry

    @abc.abstractmethod
    def render(self, states: StateSignal, inputs: Signal, outputs: Signal) -> numpy.ndarray:
        """
        Render the current state of the system.

        Returns:
            numpy.ndarray: The rendered image or visualization.
        """
