import numpy

from refactor.geometry import Geometry


class BoatGeometry(Geometry):
    def __init__(self):
        super().__init__()
        self.thrust_offset      = 3.0 # Distance between CG and Thrust vector # TODO: duplicated with geometry

        self.lateral_area       = self.thrust_offset * 2
        self.frontal_area       = 0.25 * self.lateral_area
        self.length_over_all    = self.thrust_offset * 2

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
