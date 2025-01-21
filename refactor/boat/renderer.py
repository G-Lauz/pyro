from typing import overload

import numpy
import pygame

from refactor.geometry import Geometry
from refactor.kinematic import Kinematic
from refactor.renderer import Renderer


class BoatRenderer(Renderer):
    def __init__(self, screen_size=(800, 600)):
        super().__init__()
        pygame.init()
        self.screen_size = numpy.array(screen_size)
        self.screen = pygame.display.set_mode(screen_size)

        pygame.display.set_caption("Pyro - 2D Boat Simulation")

        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 20)

        self.background_color = (255, 255, 255)
        self.grid_color = (200, 200, 200)
        self.boat_color = (0, 0, 255)
        self.arrow_color = (255, 0, 0)
        self.trajectory_color = (0, 255, 0)

        self.grid_spacing = 1 # Grid spacing in world units

    # TODO: move transformation matrix and transformation operation to a separate module
    def _translation_matrix(self, offset):
        """
        Compute the translation matrix.

        :param offset: The x, y offset. (2, 1)

        :return: The translation matrix. (3, 3)
        """
        return numpy.array([
            [1, 0, offset[0]],
            [0, 1, offset[1]],
            [0, 0, 1]
        ])
    
    def _rotation_matrix(self, angle):
        """
        Compute the rotation matrix.

        :param angle: The angle of rotation.

        :return: The rotation matrix. (3, 3)
        """
        return numpy.array([
            [numpy.cos(angle), -numpy.sin(angle), 0],
            [numpy.sin(angle), numpy.cos(angle), 0],
            [0, 0, 1]
        ])

    @overload
    def _translate_rotate_matrix(self, kinematic: Kinematic) -> numpy.ndarray:
        """
        Compute the transformation matrix to translate then rotate the boat.

        :param kinematic: The kinematic of the boat.

        :return: The transformation matrix. (3, 3)
        """
        pass

    @overload
    def _translate_rotate_matrix(self, positions: numpy.ndarray, angle: float=None) -> numpy.ndarray:
        """
        Compute the transformation matrix to translate then rotate the boat.

        :param positions: The positions of the boat. (2, 1)
        :param angle: The heading angle of the boat.

        :return: The transformation matrix. (3, 3)
        """
        pass

    def _translate_rotate_matrix(self, arg1, angle=None) -> numpy.ndarray:
        """
        Compute the transformation matrix to translate then rotate the boat.
        """
        if isinstance(arg1, Kinematic):
            positions = arg1.positions
            angle = positions[2]
        elif isinstance(arg1, numpy.ndarray) and angle is not None:
            positions = arg1
        else:
            raise TypeError("Invalid arguments. Accepts either a Kinematic object or a numpy array and an angle.")

        return numpy.array([
            [numpy.cos(angle),  -numpy.sin(angle),  positions[0]],
            [numpy.sin(angle),  numpy.cos(angle),   positions[1]],
            [0,                 0,                  1           ]
        ])
    
    def _scale_center_matrix(self, scale, offset):
        """
        Compute the transformation matrix to scale and center the boat to the screen.

        :param scale: The scale factor.
        :param offset: The x, y offset. (2, 1)
        """
        return numpy.array([
            [scale, 0,      offset[0]   ],
            [0,     scale,  offset[1]   ],
            [0,     0,      1           ]
        ])

    def _transform_points(self, transformation_matrix, points):
        """
        Apply a transformation matrix to a set of points

        :param transformation_matrix: The transformation matrix. (3, 3)
        :param points: The points to transform. (n, 2)

        :return: The transformed points. (n, 2)
        """
        assert points.shape[1] == 2 or points.shape[1] == 3, "Points must be 2D or 3D"

        isnt_homogeneous = points.shape[1] == 2

        if isnt_homogeneous: # Convert to homogeneous coordinates
            points = numpy.hstack((points, numpy.ones((points.shape[0], 1))))

        return (transformation_matrix @ points.T).T[:, :2]
    
    def _get_grid_dimensions(self, domain, scale=1, offset=numpy.array([0, 0])):
        domain_range = numpy.diff(domain, axis=1).reshape(-1)

        diagonal = numpy.linalg.norm(self.screen_size)
        difference = diagonal - domain_range

        start = numpy.zeros_like(self.screen_size) - (difference / 2) + (offset * scale)
        end = self.screen_size + (difference / 2) + (offset * scale)
        return start, end

    def _get_grid_points(self, start, end, scale=1):

        grid_spacing = self.grid_spacing * scale # Grid spacing in world units

        # Grid lines decomposed in points coordinates
        vertical_lines = numpy.arange(start[0], end[0], grid_spacing)
        horizontal_lines = numpy.arange(start[1], end[1], grid_spacing)

        horizontal_start = numpy.stack((numpy.full_like(horizontal_lines, start[0]), horizontal_lines), axis=1)
        horizontal_end = numpy.stack((numpy.full_like(horizontal_lines, end[0]), horizontal_lines), axis=1)

        vertical_start = numpy.stack((vertical_lines, numpy.full_like(vertical_lines, start[1])), axis=1)
        vertical_end = numpy.stack((vertical_lines, numpy.full_like(vertical_lines, end[1])), axis=1)

        return (
            numpy.concatenate((vertical_start, vertical_end, horizontal_start, horizontal_end), axis=0),
            vertical_lines.shape[0],
            horizontal_lines.shape[0]
        )
    
    def _draw_grid(self, grid_points, vertical_len, horizontal_len):
        for i in range(vertical_len):
            start_idx = i
            end_idx = i + vertical_len
            pygame.draw.line(self.screen, self.grid_color, grid_points[start_idx], grid_points[end_idx], 1)

        for i in range(horizontal_len):
            start_idx = 2 * vertical_len + i
            end_idx = 2 * vertical_len + i + horizontal_len
            pygame.draw.line(self.screen, self.grid_color, grid_points[start_idx], grid_points[end_idx], 1)
    
    def get_scale_and_offset(self, domain):
        """
        Compute the scale and offset to center and scale the boat to the screen.

        :param domain: The domain of the boat. (2, 2)

        :return: A tuple containing the scale and the offset. (scale (1), offset (2,1))
        """
        domain_range = numpy.diff(domain, axis=1).reshape(-1)
        scale = min(self.screen_size / domain_range)
        offset = self.screen_size // 2

        return scale, offset
    
    def arrow_points(self, input_force):
        """
        Compute the points of the arrow representing the input force.

        :param input_force: The input force. (2, 1)

        :return: The points of the arrow. (6, 3)
        """
        force_magnitude = numpy.linalg.norm(input_force) / 1000 # TODO: fix rendering patch that scale to kN
        tip_length = 0.15 * force_magnitude

        return numpy.array([
            [0, 0, 1],
            [-force_magnitude, 0, 1],
            [0, 0, 1],
            [-tip_length, tip_length, 1],
            [0, 0, 1],
            [-tip_length, -tip_length, 1]
        ])
    
    def render_info(self, kinematic: Kinematic, input_force):
        """
        Render the information about the boat on the screen.

        :param positions: The positions of the boat. (3, 1)
        :param velocities: The velocities of the boat. (3, 1)
        :param input_force: The input force of the boat. (2, 1)
        """
        positions = kinematic.positions
        velocities = kinematic.velocities
        heading = numpy.degrees(positions[2]) % 360

        numpy.set_printoptions(precision=2, suppress=True)

        text = self.font.render(f"Position: {positions[:2]} Heading: {heading:.2f}°", True, (0, 0, 0))
        self.screen.blit(text, (10, 10))

        text = self.font.render(f"Velocity: {velocities}", True, (0, 0, 0))
        self.screen.blit(text, (10, 40))

        text = self.font.render(f"Force: {input_force}", True, (0, 0, 0))
        self.screen.blit(text, (10, 70))


class FixedCameraBoatRenderer(BoatRenderer):
    def __init__(self, screen_size=(800, 600)):
        super().__init__(screen_size=screen_size)

    def _points_to_screen(self, domain, points: numpy.ndarray):
        scale, offset = self.get_scale_and_offset(domain)
        transformation_matrix = self._scale_center_matrix(scale, offset)
        return self._transform_points(transformation_matrix, points).astype(int)
        
    def render_grid(self, domain):
        scale, offset = self.get_scale_and_offset(domain)
        start, end = self._get_grid_dimensions(domain)
        grid_points, vertical_len, horizontal_len = self._get_grid_points(start, end, scale)
        self._draw_grid(grid_points, vertical_len, horizontal_len)

    def get_vector_force(self, input_force, geometry: Geometry):
        force_direction = numpy.arctan2(input_force[1], input_force[0])

        arrow_points = self.arrow_points(input_force)

        offset = numpy.array([-geometry.thrust_offset, 0])
        transformation_matrix = self._translate_rotate_matrix(offset, force_direction)
        return self._transform_points(transformation_matrix, arrow_points)

    def render(self, geometry: Geometry, kinematic: Kinematic, input_force: numpy.ndarray, trajectory: numpy.ndarray = None):
        self.screen.fill(self.background_color)

        domain = kinematic.get_domain(dynamic=True)[:2] # get x, y domain

        self.render_grid(domain)
        self.render_info(kinematic, input_force)

        transformation_matrix = self._translate_rotate_matrix(kinematic)

        # Draw the boat
        boat_shape = geometry.shape
        boat_shape = self._transform_points(transformation_matrix, boat_shape)
        boat_points = self._points_to_screen(domain, boat_shape)
        pygame.draw.polygon(self.screen, self.boat_color, boat_points)

        # Draw the input force arrow
        arrow_points = self.get_vector_force(input_force, geometry)
        arrow_points = self._transform_points(transformation_matrix, arrow_points)
        arrow_points = self._points_to_screen(domain, arrow_points)
        pygame.draw.lines(self.screen, self.arrow_color, True, arrow_points, 3)

        pygame.display.flip()


class DynamicCameraBoatRenderer(BoatRenderer):
    def __init__(self, screen_size=(800, 600)):
        super().__init__(screen_size=screen_size)

        self.heading_offset = -numpy.pi / 2 # Account for the boat heading facing the top of the screen

    def render_boat(self, domain, boat_points):
        scale, offset = self.get_scale_and_offset(domain)

        # Make the boat heading facing the top of the screen
        transformation_matrix = self._rotation_matrix(self.heading_offset)
        boat_points = self._transform_points(transformation_matrix, boat_points)

        # Scale the boat to the screen size and center it
        points = boat_points * scale + offset

        pygame.draw.polygon(self.screen, self.boat_color, points)

    def render_force(self, domain, input_force, geometry):
        force_direction = numpy.arctan2(input_force[1], input_force[0]) + self.heading_offset

        arrow_points = self.arrow_points(input_force)

        scale, offset = self.get_scale_and_offset(domain)
        offset = offset + numpy.array([0, geometry.thrust_offset]) * scale

        # Rotate the arrow based on the boat heading
        transformation_matrix = self._rotation_matrix(force_direction)
        arrow_points = self._transform_points(transformation_matrix, arrow_points)

        # Scale the arrow to the screen size and center it
        points = arrow_points * scale + offset

        pygame.draw.lines(self.screen, self.arrow_color, True, points, width=3)

    def render_grid(self, domain, kinematic: Kinematic):
        # Scale en center the grid
        scale, offset = self.get_scale_and_offset(domain)
        positions = kinematic.positions[:2]

        start, end = self._get_grid_dimensions(domain, scale, positions)
        points, vertical_len, horizontal_len = self._get_grid_points(start, end, scale)

        # Rotate and center the grid to match the boat heading
        angle = -kinematic.positions[2] + numpy.pi / 2
        x_offset = offset[0] - numpy.cos(angle) * offset[0] + numpy.sin(angle) * offset[1]
        y_offset = offset[1] - numpy.sin(angle) * offset[0] - numpy.cos(angle) * offset[1]
        offset = numpy.array([x_offset, y_offset])

        transformation_matrix = self._translate_rotate_matrix(offset, angle)
        points = self._transform_points(transformation_matrix, points)

        # Draw the grid
        self._draw_grid(points, vertical_len, horizontal_len)


    def render_trajectory(self, domain, trajectory, kinematic: Kinematic):
        # Scale en center the trajectory
        scale, offset = self.get_scale_and_offset(domain)

        position = kinematic.positions[:2]

        # Trajectory points decomposed in points coordinates
        trajectory_points = trajectory * scale + offset + position * scale

        # Rotate and center the grid to match the boat heading
        angle = -kinematic.positions[2] + numpy.pi / 2
        x_offset = offset[0] - numpy.cos(angle) * offset[0] + numpy.sin(angle) * offset[1]
        y_offset = offset[1] - numpy.sin(angle) * offset[0] - numpy.cos(angle) * offset[1]
        offset = numpy.array([x_offset, y_offset])

        transformation_matrix = self._translate_rotate_matrix(offset, angle)
        points = self._transform_points(transformation_matrix, trajectory_points)

        # draw the trajectory
        pygame.draw.lines(self.screen, self.trajectory_color, False, points, width=5)

    def render(self, geometry: Geometry, kinematic: Kinematic, input_force: numpy.ndarray, trajectory):
        self.screen.fill(self.background_color)

        domain = kinematic.get_domain(dynamic=True)[:2] # get x, y domain

        self.render_grid(domain, kinematic)
        self.render_trajectory(domain, trajectory, kinematic)
        self.render_info(kinematic, input_force)
        self.render_boat(domain, geometry.shape)
        self.render_force(domain, input_force, geometry)
        
        pygame.display.flip()
