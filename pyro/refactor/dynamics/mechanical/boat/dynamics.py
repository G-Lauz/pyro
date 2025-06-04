import numpy

from pyro.refactor.signal import Signal
from pyro.refactor.dynamics.mechanical.dynamics import MechanicalDynamics
from pyro.refactor.dynamics.mechanical.signal import MechanicalStateSignal

from .configuration import BoatConfiguration


class BoatDynamics(MechanicalDynamics):
    def __init__(self, config: BoatConfiguration):
        super().__init__()

        self.dof = config.mechanical.dof
        self.mass = config.mechanical.mass
        self.inertia = config.mechanical.inertia

        self.linear_damping = config.hydrodynamics.linear_damping
        self.water_density = config.hydrodynamics.water_density

        self.Cx_max = config.hydrodynamics.quadratic_damping.cx_max
        self.Cy_max = config.hydrodynamics.quadratic_damping.cy_max
        self.Cm_max = config.hydrodynamics.quadratic_damping.cm_max

        self.frontal_area = config.geometry.frontal_area
        self.lateral_area = config.geometry.lateral_area
        self.length_over_all = config.geometry.length_overall
        self.thrust_offset = config.geometry.thrust_offset

    def inertia_matrix(self, states: MechanicalStateSignal):
        return numpy.diag([self.mass, self.mass, self.inertia])

    def coriolis_matrix(self, states: MechanicalStateSignal):
        coriolis_matrix = numpy.zeros((self.dof, self.dof))

        angular_velocity = states.velocity[-1]  # Assuming the last velocity is the angular velocity

        coriolis_matrix[1, 0] = + self.mass * angular_velocity
        coriolis_matrix[0, 1] = - self.mass * angular_velocity
        return coriolis_matrix

    def gravitational_force(self, states: MechanicalStateSignal):
        return numpy.zeros(self.dof)

    def dissipative_forces(self, states: MechanicalStateSignal):
        # Linear damping
        linear_damping = states.velocity * self.linear_damping

        dx = states.velocity[0]
        dy = states.velocity[1]

        squared_relative_speed = dx**2 + dy**2
        direction = -numpy.arctan2(dy, dx)

        Cx, Cy, Cm = self.current_coefficients(direction)

        # Quadratic damping
        fx = -0.5 * self.water_density * self.frontal_area * Cx * squared_relative_speed
        fy = -0.5 * self.water_density * self.lateral_area * Cy * squared_relative_speed
        mz = -0.5 * self.water_density * self.lateral_area * self.length_over_all * Cm * squared_relative_speed

        quadratic_damping = numpy.array([fx, fy, mz])

        return linear_damping + quadratic_damping

    def actuators_matrix(self, states: MechanicalStateSignal, inputs: Signal):
        actuator_matrix = numpy.zeros((self.dof, inputs.dim))

        actuator_matrix[0, 0] = 1
        actuator_matrix[1, 1] = 1
        actuator_matrix[2, 1] = -self.thrust_offset

        return actuator_matrix
    
    def current_coefficients(self, direction):
        Cx = - self.Cx_max * numpy.cos(direction) * numpy.abs(numpy.cos(direction))
        Cy = + self.Cy_max * numpy.sin(direction) * numpy.abs(numpy.sin(direction))
        Cm = + self.Cm_max * numpy.sin(2.0 * direction)

        return Cx, Cy, Cm
