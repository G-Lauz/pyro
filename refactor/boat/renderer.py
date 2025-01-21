import numpy
import pygame

from refactor.geometry import Geometry
from refactor.kinematic import Kinematic
from refactor.renderer import Renderer


class FixedCameraBoatRenderer(Renderer):
    def __init__(self, screen_size=(800, 600)):
        super().__init__()
        self.pygame = pygame
        self.screen_size = screen_size
        self.screen = pygame.display.set_mode(screen_size)

        self.pygame.display.set_caption("Pyro")

        self.pygame.font.init()
        self.font = pygame.font.SysFont("Futura", 30)

    def transform(self, points, position):
        """
        Rotate and translate
        """
        transformation_matrix = numpy.array([
            [numpy.cos(position[2]),    -numpy.sin(position[2]),    position[0] ],
            [numpy.sin(position[2]),    numpy.cos(position[2]),     position[1] ],
            [0,                         0,                          1           ]
        ])

        homogeneous_points = numpy.hstack((points, numpy.ones((points.shape[0], 1))))
        rotated_points = (transformation_matrix @ homogeneous_points.T).T[:, :2]
        return rotated_points

    def _points_to_screen(self, domain, points: numpy.ndarray):
        domain_range = numpy.diff(domain, axis=1).reshape(-1)

        scale = min(self.screen_size / domain_range)
        offset = numpy.array(self.screen_size) // 2

        transformation_matrix = numpy.array([
            [scale, 0,      offset[0]   ],
            [0,     scale,  offset[1]   ],
            [0,     0,      1           ]
        ])
        homogenous_points = numpy.hstack((points, numpy.ones((points.shape[0], 1))))
        points = (transformation_matrix @ homogenous_points.T).T[:, :2]
        return points.astype(int)
        
    def _draw_grid(self, camera_offset=(0, 0)):
        grid_color = (0, 0, 0)
        grid_size = 50

        start_x = int((camera_offset[0]) // grid_size) * grid_size
        start_y = int((camera_offset[1]) // grid_size) * grid_size
        end_x = int(camera_offset[0] + self.screen_size[0]) + grid_size
        end_y = int(camera_offset[1] + self.screen_size[1]) + grid_size

        for x in range(start_x, end_x, grid_size):
            pygame.draw.line(
                self.screen, grid_color,
                (x - camera_offset[0], 0),
                (x - camera_offset[0], self.screen_size[1]),
                1
            )

        for y in range(start_y, end_y, grid_size):
            pygame.draw.line(
                self.screen, grid_color,
                (0, y - camera_offset[1]),
                (self.screen_size[0], y - camera_offset[1]),
                1
            )

    def get_vector_force(self, input_force, geometry: Geometry):
        force_magnitude = numpy.linalg.norm(input_force) / 1000 # TODO: fix rendering patch that scale to kN

        force_direction = numpy.arctan2(input_force[1], input_force[0]) #+ numpy.pi/2 # visual correction of the angles, doesn't match the simulation
        tip_length = 0.15 * force_magnitude

        arrow_points = numpy.array([
            [0, 0, 1],
            [-force_magnitude, 0, 1],
            [0, 0, 1],
            [-tip_length, tip_length, 1],
            [0, 0, 1],
            [-tip_length, -tip_length, 1]
        ])

        T = numpy.array([
            [numpy.cos(force_direction), -numpy.sin(force_direction), -geometry.thrust_offset],
            [numpy.sin(force_direction), numpy.cos(force_direction), 0],
            [0, 0, 1]
        ])
        arrow_points = (T @ arrow_points.T).T[:, :2]
        return arrow_points

    # def render(self, geometry: Geometry, kinematic: Kinematic):
    def render(self, geometry: Geometry, kinematic: Kinematic, input_force: numpy.ndarray):
        self.screen.fill((255, 255, 255))
        self._draw_grid()

        positions = kinematic.positions
        velocities = kinematic.velocities

        states_upper_bound = kinematic.states_upper_bound
        states_lower_bound = kinematic.states_lower_bound
        domain = numpy.stack((states_lower_bound[:2], states_upper_bound[:2]), axis=1)
        domain = domain + numpy.array([[positions[0]],[positions[1]]]) # dynamic domain

        # States
        text = self.font.render(f"Position: {positions}", True, (0, 0, 0))
        self.screen.blit(text, (10, 10))

        text = self.font.render(f"Velocity: {velocities}", True, (0, 0, 0))
        self.screen.blit(text, (10, 40))

        text = self.font.render(f"Force: {input_force}", True, (0, 0, 0))
        self.screen.blit(text, (10, 70))

        # positions = numpy.array([3, 3, -numpy.pi/4])

        # Draw the boat 
        boat_shape = geometry.shape
        boat_shape = self.transform(boat_shape, positions)
        boat_points = self._points_to_screen(domain, boat_shape)
        pygame.draw.polygon(self.screen, (0, 0, 255), boat_points)

        # Draw the input force arrow
        arrow_points = self.get_vector_force(input_force, geometry)
        arrow_points = self.transform(arrow_points, positions)
        arrow_points = self._points_to_screen(domain, arrow_points)
        pygame.draw.lines(self.screen, (255, 0, 0), True, arrow_points, 3)

        self.pygame.display.flip()


class DynamicCameraBoatRenderer(Renderer):
    def __init__(self, screen_size=(800, 600)):
        super().__init__()
        pygame.init()
        self.screen_size = numpy.array(screen_size)
        self.screen = pygame.display.set_mode(screen_size)

        pygame.display.set_caption("Pyro Boat Rendering")

        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 20)

    def render_info(self, positions, velocities, input_force):
        # Display position and force info
        numpy.set_printoptions(precision=2, suppress=True)
        text = self.font.render(f"Position: {positions[:2]} Heading: {numpy.degrees(positions[2]):.2f}°", True, (0, 0, 0))
        self.screen.blit(text, (10, 10))

        text = self.font.render(f"Velocity: {velocities}", True, (0, 0, 0))
        self.screen.blit(text, (10, 40))

        text = self.font.render(f"Force: {input_force}", True, (0, 0, 0))
        self.screen.blit(text, (10, 70))

    def render_boat(self, domain, boat_points):
        boat_color = (0, 0, 255)

        domain_range = numpy.diff(domain, axis=1).reshape(-1)
        scale = min(self.screen_size / domain_range)
        offset = numpy.array(self.screen_size) // 2

        # Make the boat heading facing the top of the screen
        heading_offset = -numpy.pi / 2
        R = numpy.array([
            [numpy.cos(heading_offset), -numpy.sin(heading_offset), 0],
            [numpy.sin(heading_offset), numpy.cos(heading_offset), 0],
            [0, 0, 1]
        ])
        homogenous_points = numpy.hstack((boat_points, numpy.ones((boat_points.shape[0], 1))))
        boat_points = (R @ homogenous_points.T).T[:, :2]

        # Scale the boat to the screen size and center it
        points = boat_points * scale + offset

        pygame.draw.polygon(self.screen, boat_color, points)

    def render_force(self, domain, input_force, geometry, positions):
        arrow_color = (255, 0, 0)
        force_magnitude = numpy.linalg.norm(input_force) / 1000 # TODO: fix rendering patch that scale to kN

        headinf_offset = -numpy.pi / 2 # Account for the boat heading
        force_direction = numpy.arctan2(input_force[1], input_force[0]) + headinf_offset

        # Define geometry of the arrow
        tip_length = 0.15 * force_magnitude
        arrow_points = numpy.array([
            [0, 0, 1],
            [-force_magnitude, 0, 1],
            [0, 0, 1],
            [-tip_length, tip_length, 1],
            [0, 0, 1],
            [-tip_length, -tip_length, 1]
        ])

        domain_range = numpy.diff(domain, axis=1).reshape(-1)
        scale = min(self.screen_size / domain_range)
        offset = numpy.array(self.screen_size) // 2

        offset = offset + numpy.array([0, geometry.thrust_offset]) * scale

        # Rotate the arrow based on the boat heading
        R = numpy.array([
            [numpy.cos(force_direction), -numpy.sin(force_direction), 0],
            [numpy.sin(force_direction), numpy.cos(force_direction), 0],
            [0, 0, 1]
        ])
        arrow_points = (R @ arrow_points.T).T[:, :2]

        # Scale the arrow to the screen size and center it
        points = arrow_points * scale + offset

        pygame.draw.lines(self.screen, arrow_color, True, points, width=3)

    def render_grid(self, domain, kinematic: Kinematic):
        grid_color = (200, 200, 200)

        # Scale en center the grid
        domain_range = numpy.diff(domain, axis=1).reshape(-1)
        scale = min(self.screen_size / domain_range)
        offset = numpy.array(self.screen_size) // 2

        diagonal = numpy.linalg.norm(self.screen_size)
        difference = diagonal - domain_range

        position = kinematic.positions[:2]

        start = numpy.zeros_like(self.screen_size) - difference / 2 + position * scale
        end = self.screen_size + difference / 2 + position * scale

        grid_spacing = 1 * scale # Grid spacing in world units

        # Grid lines decomposed in points coordinates
        vertical_lines = numpy.arange(start[0], end[0], grid_spacing)
        horizontal_lines = numpy.arange(start[1], end[1], grid_spacing)

        vertical_start = numpy.stack((vertical_lines, numpy.full_like(vertical_lines, start[1])), axis=1)
        vertical_end = numpy.stack((vertical_lines, numpy.full_like(vertical_lines, end[1])), axis=1)

        horizontal_start = numpy.stack((numpy.full_like(horizontal_lines, start[0]), horizontal_lines), axis=1)
        horizontal_end = numpy.stack((numpy.full_like(horizontal_lines, end[0]), horizontal_lines), axis=1)

        points = numpy.concatenate((vertical_start, vertical_end, horizontal_start, horizontal_end), axis=0)
        points = numpy.hstack((points, numpy.ones((points.shape[0], 1)))) # Homogeneous coordinates

        # Rotate and center the grid to match the boat heading
        angle = -kinematic.positions[2] + numpy.pi / 2
        x_offset = offset[0] - numpy.cos(angle) * offset[0] + numpy.sin(angle) * offset[1]
        y_offset = offset[1] - numpy.sin(angle) * offset[0] - numpy.cos(angle) * offset[1]
        R = numpy.array([
            [numpy.cos(angle), -numpy.sin(angle), x_offset],
            [numpy.sin(angle), numpy.cos(angle), y_offset],
            [0, 0, 1]
        ])
        points = (R @ points.T).T[:, :2]

        # draw the grid
        for i in range(vertical_lines.shape[0]):
            start_idx = i
            end_idx = i + vertical_lines.shape[0]
            pygame.draw.line(self.screen, grid_color, points[start_idx], points[end_idx], 1)

        for i in range(horizontal_lines.shape[0]):
            start_idx = 2 * vertical_lines.shape[0] + i
            end_idx = 2 * vertical_lines.shape[0] + i + horizontal_lines.shape[0]
            pygame.draw.line(self.screen, grid_color, points[start_idx], points[end_idx], 1)

    def render_trajectory(self, domain, trajectory, kinematic: Kinematic):
        trajectory_color = (0, 255, 0)

        # Scale en center the trajectory
        domain_range = numpy.diff(domain, axis=1).reshape(-1)
        scale = min(self.screen_size / domain_range)
        offset = numpy.array(self.screen_size) // 2

        diagonal = numpy.linalg.norm(self.screen_size)
        difference = diagonal - domain_range

        position = kinematic.positions[:2]

        # Trajectory points decomposed in points coordinates
        trajectory_points = trajectory * scale + offset + position * scale
        points = numpy.hstack((trajectory_points, numpy.ones((trajectory_points.shape[0], 1)))) # Homogeneous coordinates

        # Rotate and center the grid to match the boat heading
        angle = -kinematic.positions[2] + numpy.pi / 2
        x_offset = offset[0] - numpy.cos(angle) * offset[0] + numpy.sin(angle) * offset[1]
        y_offset = offset[1] - numpy.sin(angle) * offset[0] - numpy.cos(angle) * offset[1]
        R = numpy.array([
            [numpy.cos(angle), -numpy.sin(angle), x_offset],
            [numpy.sin(angle), numpy.cos(angle), y_offset],
            [0, 0, 1]
        ])
        points = (R @ points.T).T[:, :2]

        # draw the trajectory
        pygame.draw.lines(self.screen, trajectory_color, False, points, width=5)

    def render(self, geometry: Geometry, kinematic: Kinematic, input_force: numpy.ndarray, trajectory):
        self.screen.fill((255, 255, 255))

        # Draw x, y line center on the screen (debugging)
        # pygame.draw.line(self.screen, (0, 255, 0), (0, self.screen_size[1]//2), (self.screen_size[0], self.screen_size[1]//2), 3)
        # pygame.draw.line(self.screen, (0, 255, 0), (self.screen_size[0]//2, 0), (self.screen_size[0]//2, self.screen_size[1]), 3)

        positions = kinematic.positions
        velocities = kinematic.velocities
        domain = kinematic.get_domain(dynamic=True)[:2] # get x, y domain

        states_upper_bound = kinematic.states_upper_bound
        states_lower_bound = kinematic.states_lower_bound
        domain = numpy.stack((states_lower_bound[:2], states_upper_bound[:2]), axis=1)
        domain = domain + numpy.array([[positions[0]],[positions[1]]])

        self.render_grid(domain, kinematic)
        self.render_trajectory(domain, trajectory, kinematic)
        self.render_info(positions, velocities, input_force)
        self.render_boat(domain, geometry.shape)
        self.render_force(domain, input_force, geometry, positions)
        
        pygame.display.flip()
