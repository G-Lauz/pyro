import abc

from typing import Tuple

import numpy

from pyro.refactor.geometry import Geometry
from pyro.refactor.system import System


class RenderingEngine(abc.ABC):

    @abc.abstractmethod
    def init(self, screen_size: Tuple[int, int]) -> None:
        """
        Initialize the rendering engine with the screen size.
        
        :param screen: The size of the screen as a tuple (width, height).
        """

    @abc.abstractmethod
    def clear_screen(self, color: Tuple[int, int, int]) -> None:
        """
        Clear the screen.
        """

    @abc.abstractmethod
    def draw_polygon(self, points: list, color: Tuple[int, int, int],  width: int = 0) -> None:
        """
        Draw a polygon on the screen.
        
        :param points: A list of points defining the polygon.
        :param color: The color of the polygon as an RGB tuple.
        :param width: The width of the polygon lines.
        """

    @abc.abstractmethod
    def draw_line(self, start: Tuple[int, int], end: Tuple[int, int], color: Tuple[int, int, int], width: int = 1) -> None:
        """
        Draw a line on the screen.
        
        :param start: The starting point of the line as a tuple (x, y).
        :param end: The ending point of the line as a tuple (x, y).
        :param color: The color of the line as an RGB tuple.
        :param width: The width of the line.
        """

    @abc.abstractmethod
    def draw_lines(self, points: list, color: Tuple[int, int, int], width: int = 1) -> None:
        """
        Draw lines on the screen.
        
        :param points: A list of points defining the lines.
        :param color: The color of the lines as an RGB tuple.
        :param width: The width of the lines.
        """

    @abc.abstractmethod
    def draw_text(self, text: str, position: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        """
        Draw text on the screen.
        
        :param text: The text to draw.
        :param position: The position to draw the text at.
        :param color: The color of the text as an RGB tuple.
        """

    @abc.abstractmethod
    def update_display(self) -> None:
        """
        Update the display to show the drawn elements.
        """


class Camera(abc.ABC):

    @abc.abstractmethod
    def project(self, system: System, world_pts: numpy.ndarray) -> numpy.ndarray:
        """
        Project world points to screen coordinates.
        
        :param system: The system to project the points from.
        :param world_pts: The points in world coordinates.
        :return: The projected points in screen coordinates.
        """


class Component(abc.ABC):
    @abc.abstractmethod
    def render(self, system: System, engine: RenderingEngine, camera: Camera, **kwargs) -> None:
        """
        Render the component using the rendering engine.
        
        :param system: The system to render.
        :param engine: The rendering engine to use.
        :param camera: The camera to use for projection.
        :param kwargs: Additional parameters for rendering.
        """


class Renderer(abc.ABC):
    def __init__(self, geometry: Geometry, screen_size: Tuple[int, int] = (800, 600)):
        self.geometry = geometry
        self.screen_size = numpy.array(screen_size)

    @abc.abstractmethod
    def render(self, system: System, **kwargs) -> numpy.ndarray:
        """
        Render the current state of the system.

        :param system: The system to render.
        :param kwargs: Additional parameters for rendering.

        Returns:
            numpy.ndarray: The rendered image or visualization.
        """
