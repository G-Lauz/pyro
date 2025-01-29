import numpy
import pygame

from .continuous import ContinuousSimulation
from pyro.refactor.system import MechanicalSystem
from pyro.refactor.renderer import Renderer

class PygameInteractiveSimulation(ContinuousSimulation):
    def __init__(self, system: MechanicalSystem, renderer: Renderer):
        super().__init__(system)

        self.renderer = renderer

        self.is_running = False

        self.clock = None
        pygame.joystick.init()
        self.joysticks = []

        self.input_upper_bound = system.parameters.inputs_upper_bound
        self.input_lower_bound = system.parameters.inputs_lower_bound

    def run(self, controller=None, dt=0.1, steps=1000, render=False, callback=None, trajectory_generator=None):
        self.is_running = True

        self.clock = pygame.time.Clock()
        dt = self.clock.tick(60) / 1000

        trajectory = trajectory_generator() if trajectory_generator is not None else None

        while self.is_running:
            self._event_handler()
            input_force = self._input_handler()

            next_states = self.step(input_force, dt)
            if callback is not None:
                callback(next_states)

            if render:
                self.renderer.render(self._system, input_force, trajectory=trajectory)

            dt = self.clock.tick(60) / 1000

    def _event_handler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            if event.type == pygame.JOYDEVICEADDED:
                joystick = pygame.joystick.Joystick(event.device_index)
                self.joysticks.append(joystick)

    def _input_handler(self):
        for joystick in self.joysticks:
            # XBox 360 Left Stick (left -> right: axis 0, up -> down: axis 1)
            # See https://www.pygame.org/docs/ref/joystick.html#xbox-360-controller-pygame-2-x for more information
            x_force = joystick.get_axis(0) * (self.input_upper_bound[1] - self.input_lower_bound[1]) * 0.5
            y_force = joystick.get_axis(1) * (self.input_upper_bound[0] - self.input_lower_bound[0]) * 0.5
            input_force = numpy.array([-y_force, -x_force])

        return input_force
