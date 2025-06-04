import abc

import numpy


class Geometry(abc.ABC):

    @property
    @abc.abstractmethod
    def shape(self) -> numpy.ndarray:
        pass
