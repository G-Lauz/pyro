import abc

import numpy

from pyro.refactor.dynamics.mechanical.signal import MechanicalStateSignal


class Kinematics(abc.ABC):

    @abc.abstractmethod
    def state_dimension(self):
        """
        The dimension of the state vector.

        :return: The dimension of the state vector. (int)
        """

    @abc.abstractmethod
    def transformation_matrix(self, states: MechanicalStateSignal) -> numpy.ndarray:
        """
        Compute the transformation matrix based on the current states.

        :param states: The current states of the system. (MechanicalStateSignal)

        :return: The transformation matrix. (dof, dof)
        """
