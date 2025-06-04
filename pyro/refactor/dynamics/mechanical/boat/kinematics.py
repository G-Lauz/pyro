import numpy

from pyro.refactor.dynamics.mechanical.kinematics import Kinematics
from pyro.refactor.dynamics.mechanical.signal import MechanicalStateSignal


class BoatKinematics(Kinematics):

    def state_dimension(self):
        return 6  # x, y, theta, dx, dy, dtheta  # TODO: This should be defined in the configuration

    def transformation_matrix(self, states: MechanicalStateSignal) -> numpy.ndarray:
        theta = states.position[-1]  # Assuming the last position is the angle

        return numpy.array([
            [numpy.cos(theta),  -numpy.sin(theta),  0],
            [numpy.sin(theta),  numpy.cos(theta),   0],
            [0,                 0,                  1]
        ])
