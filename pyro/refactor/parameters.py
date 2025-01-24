import abc

import numpy


class MechanicalSystemParameters(abc.ABC):
    dof: int

    inputs_upper_bound: numpy.ndarray
    inputs_lower_bound: numpy.ndarray

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def inertia_matrix(self, *args, **kwargs):
        """
        Compute the inertia matrix of the system based on the current positions.

        Such that:

        \equation{begin}
            E_k = \frac{1}{2} \dot{q}^T H(q) \dot{q}
        \equation{end}

        Where $q$ is the position vector of the system and $H(q)$ is the inertia matrix.

        :param positions: The current positions of the system.   (dof, 1)

        :return: The inertia matrix of the system.               (dof, dof)
        """
        pass

    @abc.abstractmethod
    def coriolis_matrix(self, *args, **kwargs):
        """
        Compute the coriolis matrix of the system based on the current positions and velocities.

        Such that: d H / dt =  C + C^T

        :param positions: The current positions of the system.   (dof, 1)
        :param velocities: The current velocities of the system. (dof, 1)

        :return: The coriolis matrix of the system.              (dof, dof)
        """
        pass

    @abc.abstractmethod
    def gravitational_force(self, *args, **kwargs):
        """
        Compute the gravitational force of the system.

        :return: The gravitational force of the system.   (dof, 1)
        """
        pass

    @abc.abstractmethod
    def dissipative_forces(self, *args, **kwargs):
        """
        Compute the dissipative forces of the system.

        :param positions: The current positions of the system.      (dof, 1)
        :param velocities: The current velocities of the system.    (dof, 1)

        :return: The dissipative forces of the system.              (dof, 1)
        """
        pass

    @abc.abstractmethod
    def actuators_matrix(self, *args, **kwargs):
        """
        Compute the actuator matrix of the system based on the current positions.

        :param positions: The current positions of the system.   (dof, 1)

        :return: The actuator matrix of the system.              (dof, n_actuators)
        """
        pass
