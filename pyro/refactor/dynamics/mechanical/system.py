from typing import Dict, Optional

import numpy

from pyro.refactor.system import DynamicSystem

from .kinematics import Kinematics
from .signal import MechanicalStateSignal
from .dynamics import MechanicalDynamics


class MechanicalSystem(DynamicSystem):
    def __init__(self, name: str, kinematics: Kinematics, dynamics: MechanicalDynamics, initial_states: Optional[numpy.ndarray] = None):

        states_dim = kinematics.state_dimension()  # TODO: This should be defined in the configuration

        # TODO: We could expose those signal as parameters of the constructor
        # so the user can set them up as he wants
        state_signal = MechanicalStateSignal(name="x",
                              dim=states_dim,
                              initial_values= initial_states if initial_states is not None else numpy.zeros(states_dim),
                              lower_bounds=numpy.full(states_dim, -numpy.inf),
                              upper_bounds=numpy.full(states_dim, numpy.inf))

        super().__init__(name=name, state_signal=state_signal)

        self.kinematics = kinematics
        self.dynamics = dynamics

    @property
    def positions(self) -> numpy.ndarray:
        """
        Get the position values from the state signal.

        :return: The position values of the system.
        """
        state_signal: MechanicalStateSignal = self.state_signal
        return state_signal.position

    @positions.setter
    def positions(self, values: numpy.ndarray):
        """
        Set the position values in the state signal.

        :param values: The position values to set.
        """
        state_signal: MechanicalStateSignal = self.state_signal
        state_signal.position = values

    @property
    def velocities(self) -> numpy.ndarray:
        """
        Get the velocity values from the state signal.

        :return: The velocity values of the system.
        """
        state_signal: MechanicalStateSignal = self.state_signal
        return state_signal.velocity

    @velocities.setter
    def velocities(self, values: numpy.ndarray):
        """
        Set the velocity values in the state signal.

        :param values: The velocity values to set.
        """
        state_signal: MechanicalStateSignal = self.state_signal
        state_signal.velocity = values

    def acceleration(self):
        H = self.dynamics.inertia_matrix(self.state_signal)
        C = self.dynamics.coriolis_matrix(self.state_signal)
        g = self.dynamics.gravitational_force(self.state_signal)
        d = self.dynamics.dissipative_forces(self.state_signal)
        B = self.dynamics.actuators_matrix(self.state_signal, self.inputs["u"]) # TODO: better inputs interface

        velocities = self.velocities
        inputs = self.inputs["u"].values

        accelerations = numpy.linalg.inv(H) @ (B @ inputs - C @ velocities - g - d)

        return accelerations

    def compute_dynamics(self, time: float = 1.0, dt: float = 0.01) -> numpy.ndarray:
        velocities = self.kinematics.transformation_matrix(self.state_signal) @ self.velocities # apply transformation matrix from kinematics
        accelerations = self.acceleration()

        states_derivative = numpy.concatenate((velocities, accelerations), axis=0)
        return states_derivative

    def compute_output(self, time: float, dt: float = 0.01) -> Dict[str, numpy.ndarray]:
        return {
            "y": self.states
        }
