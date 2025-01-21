import abc

import numpy


class Kinematic(abc.ABC):

    positions: numpy.ndarray # q
    velocities: numpy.ndarray # dq

    states_upper_bound: numpy.ndarray
    states_lower_bound: numpy.ndarray

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

    @abc.abstractmethod
    def get_domain(self, dynamic=False):
        """
        Get the domain of the system.

        :param dynamic: Whether the domain should be dynamic.

        :return: The domain of the system.
        """
        return numpy.stack((self.states_lower_bound, self.states_upper_bound), axis=1)

    @abc.abstractmethod
    def _transformation_matrix(self):
        """
        Compute the transformation matrix from generalized velocities to derivatives of
        configuration variables.

        :param configuration: The configuration of the system.

        :return: The transformation matrix.   (n_configurations, dof)
        """
        return numpy.eye(len(self.properties.configuration), self.properties.dof)
    
    @abc.abstractmethod
    def update_states(self, dv, dt=0.01):
        """
        Update the state of the system based on the acceleration.

        :param dv: The acceleration of the system.    (dof, 1)
        :param dt: The time step.                     (1, 1)
        """
        dq = self._transformation_matrix() @ self.velocities
        
        self.positions += dq * dt
        self.velocities += dv * dt


