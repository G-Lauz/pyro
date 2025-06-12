from typing import Tuple

import numpy

from pyro.refactor.geometry import Geometry
from pyro.refactor.signal import Signal
from pyro.refactor.dynamics.mechanical.system import MechanicalSystem
from pyro.refactor.dynamics.mechanical.signal import MechanicalStateSignal
from pyro.refactor.renderer import Camera, Component, Renderer, PygameRenderingEngine
from pyro.refactor.utils import Transformation2D


class BoatFrameCamera(Camera):
    def __init__(self, screen_size: Tuple[int, int] = (800, 600)):
        self.screen_size = screen_size
        
        # To rotate the world towards the top of the screen
        self.heading_offset = -numpy.pi / 2

    def apply_reference_frame(self, points: numpy.ndarray, positions: numpy.ndarray, scale, offset) -> numpy.ndarray:
        """
        This frame is relative to the center of the screen with a rotation making the x-axis point upwards.
        """
        # Rotate the world points to face the top of the screen
        rotation_matrix = Transformation2D.rotation_matrix(self.heading_offset)
        rotated_pts = Transformation2D.transform_points(rotation_matrix, points)

        # Scale and translate the points to fit the screen
        screen_pts = rotated_pts * scale + offset

        return screen_pts
    
    def apply_world_frame(self, points: numpy.ndarray, positions: numpy.ndarray, scale, offset) -> numpy.ndarray:
        """
        This frame is relative to the world coordinates, translation and rotation are applied based on the boat's position and heading.
        """
        rotation_matrix = Transformation2D.rotation_matrix(self.heading_offset - positions[2])
        rotated_pts = Transformation2D.transform_points(rotation_matrix, points)

        translation = Transformation2D.transform_points(rotation_matrix, positions[:2].reshape(1, -1)).reshape(-1)

        translation_matrix = Transformation2D.translation_matrix(-translation)
        transformed_pts = Transformation2D.transform_points(translation_matrix, rotated_pts)

        screen_pts = transformed_pts * scale + offset

        return screen_pts

    def project(self, system: MechanicalSystem, points: numpy.ndarray, frame="world") -> numpy.ndarray:
        """
        Project world points to screen coordinates based on the boat's position and orientation.
        """
        state_signal: MechanicalStateSignal = system.state_signal

        positions = state_signal.position
        velocities = state_signal.velocity

        domain = state_signal.get_domain()
        domain = domain + numpy.array([*positions, *velocities]).reshape(-1, 1) # Center the domain on the boat position
        domain = domain[:2] # Get (x, y) domain

        domain_range = numpy.diff(domain, axis=1).reshape(-1)
        scale = min(self.screen_size / domain_range)
        offset = self.screen_size // 2

        if frame == "reference":
            screen_pts = self.apply_reference_frame(points, positions, scale, offset)
        elif frame == "world":
            screen_pts = self.apply_world_frame(points, positions, scale, offset)
        else:
            raise ValueError(f"Unknown frame type: {frame}. Use 'reference' or 'world'.")

        return screen_pts


class BoatComponent(Component):
    def __init__(self, geometry: Geometry):
        super().__init__()
        self.geometry = geometry

        self.boat_color = (0, 0, 255)  # Blue color for the boat

    def render(self, system: MechanicalSystem, engine: PygameRenderingEngine, camera: Camera, **kwargs):
        world_points = self.geometry.shape
        screen_points = camera.project(system, world_points, frame="reference")
        engine.draw_polygon(screen_points, self.boat_color)


class GridComponent(Component):
    def __init__(self, screen_size: Tuple[int,int], grid_units: int = 1):
        super().__init__()
        self.screen_size = screen_size
        self.grid_units = grid_units

        self.grid_color = (200, 200, 200)  # Light gray color for the grid

    def render(self, system: MechanicalSystem, engine: PygameRenderingEngine, camera: Camera, **kwargs):
        system_state: MechanicalStateSignal = system.state_signal

        positions = system_state.position
        velocities = system_state.velocity

        # Calculate visible screen area in world coordinates
        screen_width, screen_height = self.screen_size

        # Calculate scale factor
        domain = system_state.get_domain()
        domain = domain + numpy.array([*positions, *velocities]).reshape(-1, 1) # Center the domain on the boat position
        domain = domain[:2] # Get (x, y) domain

        domain_range = numpy.diff(domain, axis=1).reshape(-1)
        scale = min(self.screen_size / domain_range)

        # Calculate grid parameters with buffer zone (1.5x screen size)
        buffer_factor = 1.5
        grid_spacing = self.grid_units * scale

        # Calculate grid bounds with buffer
        x_buffer = buffer_factor * screen_width
        y_buffer = buffer_factor * screen_height

        # Calculate modulo offset to ensure grid lines move smoothly with boat
        x_mod_offset = (positions[0] * scale) % grid_spacing
        y_mod_offset = (positions[1] * scale) % grid_spacing

        # Generate grid lines
        vertical_lines = numpy.arange(-x_buffer/2, x_buffer/2 + grid_spacing, grid_spacing) - x_mod_offset
        horizontal_lines = numpy.arange(-y_buffer/2, y_buffer/2 + grid_spacing, grid_spacing) - y_mod_offset

        # Create line endpoints
        horizontal_start = numpy.stack((numpy.full_like(horizontal_lines, -x_buffer/2), horizontal_lines), axis=1)
        horizontal_end = numpy.stack((numpy.full_like(horizontal_lines, x_buffer/2), horizontal_lines), axis=1)

        vertical_start = numpy.stack((vertical_lines, numpy.full_like(vertical_lines, -y_buffer/2)), axis=1)
        vertical_end = numpy.stack((vertical_lines, numpy.full_like(vertical_lines, y_buffer/2)), axis=1)

        grid_points = numpy.concatenate((vertical_start, vertical_end, horizontal_start, horizontal_end), axis=0)
        
        # Center and rotate grid
        offset = self.screen_size // 2
        heading_offset = numpy.pi / 2
        angle = -positions[2] - heading_offset

        # Transform grid points to screen coordinates
        transformation_matrix = Transformation2D.translate_rotate_matrix(offset, angle)
        grid_points = Transformation2D.transform_points(transformation_matrix, grid_points)

        # Draw grid
        vertical_len = vertical_lines.shape[0]
        for i in range(vertical_len):
            start_idx = i
            end_idx = i + vertical_len
            engine.draw_line(grid_points[start_idx], grid_points[end_idx], color=self.grid_color, width=1)

        horizontal_len = horizontal_lines.shape[0]
        for i in range(horizontal_len):
            start_idx = 2 * vertical_len + i
            end_idx = 2 * vertical_len + i + horizontal_len
            engine.draw_line(grid_points[start_idx], grid_points[end_idx], color=self.grid_color, width=1)


class InfoComponent(Component):
    def __init__(self):
        super().__init__()

        self.text_color = (0, 0, 0)  # Black color for the text

    def render(self, system: MechanicalSystem, engine: PygameRenderingEngine, camera: Camera, **kwargs):
        state_signal: MechanicalStateSignal = system.state_signal
        input_signal: Signal = system.inputs["u"].values  # TODO: better interface

        position = state_signal.position[:2]
        velocity = state_signal.velocity

        heading = state_signal.position[2]
        heading = numpy.degrees(heading) % 360

        numpy.set_printoptions(precision=2, suppress=True)

        # Positions
        position_text = f"Position: {position[:2]}, Heading: {heading:.2f}°"
        engine.draw_text(position_text, (10, 10), color=self.text_color)

        # Velocities
        velocity_text = f"Velocity: {velocity[:2]}, Angular Velocity: {velocity[2]:.2f} rad/s"
        engine.draw_text(velocity_text, (10, 40), color=self.text_color)

        # Force
        force_text = f"Force: {input_signal}"
        engine.draw_text(force_text, (10, 70), color=self.text_color)


class ForceComponent(Component):
    def __init__(self, geometry: Geometry):
        super().__init__()
        self.geometry = geometry

        self.force_color = (255, 0, 0)  # Red color for the force vector

    def render(self, system: MechanicalSystem, engine: PygameRenderingEngine, camera: Camera, **kwargs):
        input_signal: Signal = system.inputs["u"].values # TODO: better interface

        force_direction = numpy.arctan2(input_signal[1], input_signal[0])  # Get the direction of the force vector

        # Get arrow points
        force_magnitude = numpy.linalg.norm(input_signal) / 1000 # TODO: fix rendering patch that scale to kN
        tip_length = 0.15 * force_magnitude

        arrow_points = numpy.array([
            [0, 0, 1],
            [-force_magnitude, 0, 1],
            [0, 0, 1],
            [-tip_length, tip_length, 1],
            [0, 0, 1],
            [-tip_length, -tip_length, 1]
        ])

        offset = numpy.array([-self.geometry.thrust_offset, 0])
        transformation_matrix = Transformation2D.translate_rotate_matrix(offset, force_direction)
        arrow_points = Transformation2D.transform_points(transformation_matrix, arrow_points)

        arrow_points = camera.project(system, arrow_points, frame="reference")

        engine.draw_lines(arrow_points, color=self.force_color, width=3)


class BoatRenderer(Renderer):
    def __init__(self, geometry: Geometry, screen_size: Tuple[int, int] = (800, 600)):
        super().__init__(geometry=geometry, screen_size=screen_size)

        self.background_color = (255, 255, 255)  # White background

        self.engine = PygameRenderingEngine(caption="Pyro - 2D Boat Simulation")
        self.engine.init(screen_size=self.screen_size)

        self.components = [
            GridComponent(screen_size=self.screen_size),
            InfoComponent(),
            BoatComponent(geometry=self.geometry),
            ForceComponent(geometry=self.geometry)
        ]
        self.camera = BoatFrameCamera(screen_size=self.screen_size)

    def add_component(self, component: Component):
        """
        Add a component to the renderer.
        
        :param component: The component to add.
        """
        if not isinstance(component, Component):
            raise TypeError("Component must be an instance of Component class.")
        self.components.append(component)

    def add_components(self, components: list):
        """
        Add multiple components to the renderer.
        
        :param components: A list of components to add.
        """
        for component in components:
            self.add_component(component)

    def add_component_at(self, index: int, component: Component):
        """
        Add a component at a specific index in the components list.
        
        :param index: The index to insert the component at.
        :param component: The component to add.
        """
        if not isinstance(component, Component):
            raise TypeError("Component must be an instance of Component class.")
        self.components.insert(index, component)

    def render(self, system: MechanicalSystem, **kwargs):
        self.engine.clear_screen(self.background_color)

        for component in self.components:
            component.render(system=system, engine=self.engine, camera=self.camera, **kwargs)

        self.engine.update_display()
