import numpy

from .configuration import Boat2DConfiguration
from .kinematic import BoatKinematic
from pyro.refactor.parameters import MechanicalSystemParameters


class BoatParameters(MechanicalSystemParameters):
    def __init__(self, config: Boat2DConfiguration):
        super().__init__()

        # Dynamic properties
        self.dof                = config.parameters.dof # x, y, theta

        self.force_inputs       = config.parameters.force_inputs # F_x, F_y
        self.mass               = config.parameters.mass
        self.inertia            = config.parameters.inertia
        self.thrust_offset      = config.geometry.thrust_offset # Distance between CG and Thrust vector

        # Inputs bounds
        self.inputs_upper_bound = numpy.array(config.parameters.input_upper_bound)
        self.inputs_lower_bound = numpy.array(config.parameters.input_lower_bound)
        
        # Hydrodynamic coefficients

        # linear damping
        self.damping_coef = numpy.array(config.parameters.linear_damping_coefficients)

        # quadratic damping
        self.Cx_max = config.parameters.quadratic_damping_coefficients.cx_max
        self.Cy_max = config.parameters.quadratic_damping_coefficients.cy_max
        self.Cm_max = config.parameters.quadratic_damping_coefficients.cm_max

        self.water_density      = config.parameters.water_density
        self.lateral_area       = config.geometry.lateral_area
        self.frontal_area       = config.geometry.frontal_area
        self.length_over_all    = config.geometry.length_overall


    def inertia_matrix(self, *args, **kwargs):
        return numpy.diag([self.mass, self.mass, self.inertia])

    def coriolis_matrix(self, kinematic: BoatKinematic, *args, **kwargs):
        coriolis_matrix = numpy.zeros((self.dof, self.dof))

        angular_velocity = kinematic.dtheta

        coriolis_matrix[1, 0] = + self.mass * angular_velocity
        coriolis_matrix[0, 1] = - self.mass * angular_velocity
        return coriolis_matrix

    def gravitational_force(self, *args, **kwargs):
        return numpy.zeros(self.dof)

    def dissipative_forces(self, kinematic: BoatKinematic, *args, **kwargs):
        # Linear damping
        linear_damping = kinematic.velocities * self.damping_coef

        squared_relative_speed = kinematic.dx**2 + kinematic.dy**2
        direction = -numpy.arctan2(kinematic.dy, kinematic.dx)

        Cx, Cy, Cm = self.current_coefficients(direction)

        # Quadratic damping
        fx = -0.5 * self.water_density * self.frontal_area * Cx * squared_relative_speed
        fy = -0.5 * self.water_density * self.lateral_area * Cy * squared_relative_speed
        mz = -0.5 * self.water_density * self.lateral_area * self.length_over_all * Cm * squared_relative_speed

        quadratic_damping = numpy.array([fx, fy, mz])

        return linear_damping + quadratic_damping

    def actuators_matrix(self, *args, **kwargs):
        actuator_matrix = numpy.zeros((self.dof, self.force_inputs))

        actuator_matrix[0, 0] = 1
        actuator_matrix[1, 1] = 1
        actuator_matrix[2, 1] = -self.thrust_offset

        return actuator_matrix
    
    def current_coefficients(self, direction, *args, **kwargs):
        Cx = - self.Cx_max * numpy.cos(direction) * numpy.abs(numpy.cos(direction))
        Cy = + self.Cy_max * numpy.sin(direction) * numpy.abs(numpy.sin(direction))
        Cm = + self.Cm_max * numpy.sin(2.0 * direction)

        return Cx, Cy, Cm
