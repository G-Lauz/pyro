import numpy

from typing import Union, List

from pyro.refactor.signal import StateSignal


class MechanicalStateSignal(StateSignal):
    def __init__(self, 
                 name: str,
                 dim: int,
                 initial_values: Union[numpy.ndarray, List[float], float],
                 lower_bounds: Union[numpy.ndarray, List[float], float],
                 upper_bounds: Union[numpy.ndarray, List[float], float]
                 ):
        super().__init__(name, dim, initial_values, lower_bounds, upper_bounds)

    @property
    def position(self) -> numpy.ndarray:
        return self.values[:self.dim // 2]
    
    @position.setter
    def position(self, values: numpy.ndarray):
        if values.shape[0] != self.dim // 2:
            raise ValueError(f"Dimension mismatch: expected {self.dim // 2}, got {values.shape[0]}")
        self.values[:self.dim // 2] = values.flatten()

    @property
    def velocity(self) -> numpy.ndarray:
        return self.values[self.dim // 2:]
    
    @velocity.setter
    def velocity(self, values: numpy.ndarray):
        if values.shape[0] != self.dim // 2:
            raise ValueError(f"Dimension mismatch: expected {self.dim // 2}, got {values.shape[0]}")
        self.values[self.dim // 2:] = values.flatten()
