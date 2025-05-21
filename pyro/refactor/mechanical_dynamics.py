import abc

from typing import Dict

from pyro.refactor.signal import Signal


class MechanicalDynamics(abc.ABC):

    @abc.abstractmethod
    def inertia_matrix(self, signals: Dict[str, Signal]):
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

    @abc.abstractmethod
    def coriolis_matrix(self, signals: Dict[str, Signal]):
        """
        Compute the coriolis matrix of the system based on the current positions and velocities.

        Such that: d H / dt =  C + C^T

        :param positions: The current positions of the system.   (dof, 1)
        :param velocities: The current velocities of the system. (dof, 1)

        :return: The coriolis matrix of the system.              (dof, dof)
        """

    @abc.abstractmethod
    def gravitational_force(self, signals: Dict[str, Signal]):
        """
        Compute the gravitational force of the system.

        :return: The gravitational force of the system.   (dof, 1)
        """

    @abc.abstractmethod
    def dissipative_forces(self, signals: Dict[str, Signal]):
        """
        Compute the dissipative forces of the system.

        :param positions: The current positions of the system.      (dof, 1)
        :param velocities: The current velocities of the system.    (dof, 1)

        :return: The dissipative forces of the system.              (dof, 1)
        """

    @abc.abstractmethod
    def actuators_matrix(self, signals: Dict[str, Signal]):
        """
        Compute the actuator matrix of the system based on the current positions.

        :param positions: The current positions of the system.   (dof, 1)

        :return: The actuator matrix of the system.              (dof, n_actuators)
        """

    @abc.abstractmethod
    def transformation_matrix(self, signals: Dict[str, Signal]):
        """
        Compute the transformation matrix from generalized velocities to derivatives of
        configuration variables.

        :param configuration: The configuration of the system.

        :return: The transformation matrix.   (n_configurations, dof)
        """
