import abc

import numpy


class Kinematics(abc.ABC):

    @property
    @abc.abstractmethod
    def states(self):
        """
        The states of the system.
        """

    @states.setter
    @abc.abstractmethod
    def states(self, states):
        """
        Set the states of the system.
        """

    @abc.abstractmethod
    def state_dimension(self):
        """
        The dimension of the state vector.

        :return: The dimension of the state vector. (int)
        """

    @abc.abstractmethod
    def transformation_matrix(self):
        """
        Compute the transformation matrix from generalized velocities to derivatives of
        configuration variables.

        :param configuration: The configuration of the system.

        :return: The transformation matrix. 
        """


class MechanicalKinematics(Kinematics):

    positions: numpy.ndarray # q
    velocities: numpy.ndarray # dq

    @property
    def states(self):
        """
        The states of the system.
        """
        return numpy.concatenate([self.positions, self.velocities])

    @states.setter
    def states(self, states):
        """
        Set the states of the system.
        """
        self.positions = states[:len(self.positions)]
        self.velocities = states[len(self.positions):]

    def state_dimension(self):
        """
        The dimension of the state vector.

        :return: The dimension of the state vector. (int)
        """
        return len(self.states)