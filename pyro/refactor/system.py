import abc

import numpy

from pyro.refactor.geometry import Geometry
from pyro.refactor.kinematic import Kinematic
from pyro.refactor.parameters import MechanicalSystemParameters


class ContinuousDynamicSystem(abc.ABC):
    
    def __init__(self, kinematic: Kinematic):
        super().__init__()
        self.kinematic = kinematic

    @abc.abstractmethod
    def compute_dynamics(self, input, dt):
        """
        Compute the dynamics of the system from the state, the control input and the time step.

        \equation{begin}
            dx = f(x, u, t)
        \equation{end}

        :param state: The state vector of the system.       (state_dimension, 1)
        :param input: The control input vector.             (input_dimension, 1)
        :param dt: The time step.                           (1, 1)

        :return: The state derivative vector.               (state_dimension, 1)
        """
        pass


class MechanicalSystem(ContinuousDynamicSystem):

    def __init__(self, parameters: MechanicalSystemParameters, kinematic: Kinematic, geometry: Geometry):
        super().__init__(kinematic=kinematic)

        self.parameters = parameters
        self.kinematic = kinematic
        self.geometry = geometry

    def accelerations(self, input):
        """
        Compute the accelerations of the system (forward dynamic) based on the actuator forces, the actual position and the actual velocities.

        :param positions: The current positions of the system.      (dof, 1)
        :param velocities: The current velocities of the system.    (dof, 1)
        :param input: The actuator forces.                          (actuator_dimension, 1)
        :param time: The current time.                              (1, 1)

        :return: The accelerations of the system.                   (dof, 1)
        """

        H = self.parameters.inertia_matrix(kinematic=self.kinematic, input=input) # TODO improve interface
        C = self.parameters.coriolis_matrix(kinematic=self.kinematic, input=input)
        g = self.parameters.gravitational_force(kinematic=self.kinematic, input=input)
        d = self.parameters.dissipative_forces(kinematic=self.kinematic, input=input)
        B = self.parameters.actuators_matrix(kinematic=self.kinematic, input=input)

        accelerations = numpy.linalg.inv(H) @ (B @ input - C @ self.kinematic.velocities - g - d)

        return accelerations

    def compute_dynamics(self, input, dt):

        # config = self.properties.configuration.from_states(state)

        # Compute derivative of position variables
        # dq = self.transformation_matrix(config.theta) @ config.dq

        # dv = self.accelerations(angle, speed, input, time)
        # dv = self.accelerations(config, input, time)

        # dstate = numpy.concatenate((dq, dv))
        # return dstate

        dv = self.accelerations(input)
        self.kinematic.update_states(dv, dt=dt)
        return self.kinematic.states
