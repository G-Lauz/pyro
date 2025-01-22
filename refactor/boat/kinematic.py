import numpy

from .configuration import Boat2DConfiguration
from refactor.kinematic import Kinematic
    

class BoatKinematic(Kinematic):
    def __init__(self, config: Boat2DConfiguration):
        super().__init__()
        self.x = config.kinematic.x
        self.y = config.kinematic.y
        self.theta = config.kinematic.theta

        self.dx = config.kinematic.dx
        self.dy = config.kinematic.dy
        self.dtheta = config.kinematic.dtheta

        self.states_upper_bound = numpy.array(config.parameters.state_upper_bound)
        self.states_lower_bound = numpy.array(config.parameters.state_lower_bound)

    @property
    def positions(self):
        return numpy.array([self.x, self.y, self.theta])

    @positions.setter
    def positions(self, value):
        self.x = value[0]
        self.y = value[1]
        self.theta = value[2]

    @property
    def velocities(self):
        return numpy.array([self.dx, self.dy, self.dtheta])

    @velocities.setter
    def velocities(self, value):
        self.dx = value[0]
        self.dy = value[1]
        self.dtheta = value[2]

    def get_domain(self, dynamic=False):
        domain = None
        if dynamic:
            domain = numpy.stack((self.states_lower_bound, self.states_upper_bound), axis=1)
            domain = domain + numpy.array([*self.positions, *self.velocities]).reshape(-1, 1)
        else:
            domain = numpy.stack((self.states_lower_bound, self.states_upper_bound), axis=1)
        return domain

    def _transformation_matrix(self):
        return numpy.array([
            [numpy.cos(self.theta), -numpy.sin(self.theta), 0],
            [numpy.sin(self.theta), +numpy.cos(self.theta), 0],
            [0,                     0,                      1]
        ])
    
    def update_states(self, dv, dt=0.01):
        super().update_states(dv, dt=dt)
