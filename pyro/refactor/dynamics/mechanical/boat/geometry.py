import numpy

from pyro.refactor.geometry import Geometry
from .configuration import BoatGeometryConfiguration


class BoatGeometry(Geometry):
    def __init__(self, config: BoatGeometryConfiguration):
        super().__init__()
        self.thrust_offset      = config.thrust_offset

        self.lateral_area       = config.lateral_area
        self.frontal_area       = config.frontal_area
        self.length_over_all    = config.length_overall

        self.width = self.frontal_area
        self.height = self.length_over_all

        self.points = numpy.zeros((6, 3))
        self.points[0, :] = numpy.array([-self.height/2, +self.width/2, 0])
        self.points[1, :] = numpy.array([-self.height/2, -self.width/2, 0])
        self.points[2, :] = numpy.array([+self.height/2, -self.width/2, 0])
        self.points[3, :] = numpy.array([self.height/2 + self.width/2, 0, 0])
        self.points[4, :] = numpy.array([+self.height/2, +self.width/2, 0])
        self.points[5, :] = numpy.array([-self.height/2, +self.width/2, 0])

    @property
    def shape(self):
        return self.points[:, :2]
