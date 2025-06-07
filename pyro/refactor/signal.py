import abc

from typing import List, Union

import numpy


class Signal(abc.ABC):
    name: str
    dim: int

    initial_values: Union[numpy.ndarray, List[float], float]
    values: Union[numpy.ndarray, List[float], float]

    lower_bounds: Union[numpy.ndarray, List[float], float]
    upper_bounds: Union[numpy.ndarray, List[float], float]

    def __init__(self, 
                 name: str,
                 dim: int,
                 initial_values: Union[numpy.ndarray, List[float], float],
                 lower_bounds: Union[numpy.ndarray, List[float], float],
                 upper_bounds: Union[numpy.ndarray, List[float], float],
                 saturation: bool = False
                 ):
        self.name = name
        self.dim = dim

        self.initial_values = initial_values
        self.values = initial_values.copy() if isinstance(initial_values, numpy.ndarray) else numpy.array(initial_values)

        self.lower_bounds = lower_bounds if isinstance(lower_bounds, numpy.ndarray) else numpy.array(lower_bounds)
        self.upper_bounds = upper_bounds if isinstance(upper_bounds, numpy.ndarray) else numpy.array(upper_bounds)

        self.saturation = saturation

    def __repr__(self):
        return f"Signal(name={self.name}, dim={self.dim}, initial_values={self.initial_values}, values={self.values}"

    def get_domain(self):
        """
        Get the domain of the signal.

        :return: The domain of the signal.          (dim, 2)
        """
        return numpy.stack((self.lower_bounds, self.upper_bounds), axis=1)

    def update(self, values: Union[numpy.ndarray, float]):
        """
        Update the signal values.

        :param values: The new values for the signal.
        """
        if isinstance(values, numpy.ndarray) and values.shape[0] != self.dim:
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {values.shape[0]}")

        if isinstance(values, float):
            values = [values]

        if isinstance(values, list):
            values = numpy.array(values)

        if self.saturation:
            values = numpy.clip(values, self.lower_bounds, self.upper_bounds)

        self.values = values.copy()

    def reset(self):
        """
        Reset the signal to its initial values.
        """
        self.values = self.initial_values.copy() if isinstance(self.initial_values, numpy.ndarray) else numpy.array(self.initial_values)


class StateSignal(Signal):
    def __init__(self, 
                 name: str,
                 dim: int,
                 initial_values: Union[numpy.ndarray, List[float], float],
                 lower_bounds: Union[numpy.ndarray, List[float], float],
                 upper_bounds: Union[numpy.ndarray, List[float], float]
                 ):
        super().__init__(name, dim, initial_values, lower_bounds, upper_bounds)

    @property
    def states(self) -> numpy.ndarray:
        """
        Get the current state values of the signal.

        :return: The current state values.          (dim, 1)
        """
        return self.values

    @states.setter
    def states(self, values: numpy.ndarray):
        """
        Set the current state values of the signal.

        :param values: The new state values.          (dim, 1)
        """
        if values.shape[0] != self.dim:
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {values.shape[0]}")
        self.update(values.flatten())
