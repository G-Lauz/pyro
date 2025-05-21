from typing import List, Union

import numpy


class Signal:
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
                 upper_bounds: Union[numpy.ndarray, List[float], float]
                 ):
        self.name = name
        self.dim = dim

        self.initial_values = initial_values
        self.values = initial_values.copy() if isinstance(initial_values, numpy.ndarray) else numpy.array(initial_values)

        self.lower_bounds = lower_bounds if isinstance(lower_bounds, numpy.ndarray) else numpy.array(lower_bounds)
        self.upper_bounds = upper_bounds if isinstance(upper_bounds, numpy.ndarray) else numpy.array(upper_bounds)

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

        self.values = values.copy()
