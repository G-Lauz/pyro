from typing import Union

import numpy

from pyro.refactor.system import System
from pyro.refactor.model import Model
from pyro.refactor.renderer import Renderer
from .continuous import ContinuousSimulation


class PygameSimulation(ContinuousSimulation):
    def __init__(self, model: Union[Model, System], renderer: Renderer):
        super().__init__(model=model)

        # Lazy import pygame
        import pygame
        self.pygame = pygame

        self.renderer = renderer

        self.is_running = False

        self.clock = None

    def _event_handler(self):
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.is_running = False

    def run(self, dt=0.1, steps=1000, render=False, callback=None):
        self.is_running = True

        current_time = 0.0
        self.clock = self.pygame.time.Clock()
        dt = self.clock.tick(60) / 1000

        while self.is_running:
            self._event_handler()

            self._model.inputs["u"].update([10000.0, 0.0])  # TODO: Remove this mock input handling
            self._model.step(time=current_time, dt=dt)

            if callback is not None:
                raise NotImplementedError("Callback is not implemented in PygameSimulation")
                callback()

            if render:
                # TODO: won't work with Model, only with System
                self.renderer.render(self._model.state_signal, self._model.inputs["u"], self._model.outputs["y"])

            dt = self.clock.tick(60) / 1000
            current_time += dt


class PygameInteractiveSimulation(ContinuousSimulation):
    def __init__(self, system: System, renderer: Renderer):
        super().__init__(model=system)

        # Lazy import pygame
        import pygame
        self.pygame = pygame

        self.renderer = renderer

        self.is_running = False

        self.clock = None
        self.pygame.joystick.init()
        self.joysticks = []

    def run(self, dt=0.1, steps=1000, render=False, callback=None):
        self.is_running = True

        current_time = 0.0
        self.clock = self.pygame.time.Clock()
        dt = self.clock.tick(60) / 1000

        while self.is_running:
            self._event_handler()
            input_force = self._input_handler()

            self._model.inputs["u"].update(input_force)  # TODO: better interface for inputs
            self._model.step(time=current_time, dt=dt)

            if callback is not None:
                raise NotImplementedError("Callback is not implemented in PygameInteractiveSimulation")
                callback()

            if render:
                # TODO: won't work with Model, only with System
                self.renderer.render(self._model.state_signal, self._model.inputs["u"], self._model.outputs["y"])

            dt = self.clock.tick(60) / 1000
            current_time += dt

    def _event_handler(self):
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.is_running = False
            if event.type == self.pygame.JOYDEVICEADDED:
                joystick = self.pygame.joystick.Joystick(event.device_index)
                self.joysticks.append(joystick)

    def _input_handler(self):
        input_upper_bound = self._model.inputs["u"].upper_bounds  # TODO: better interface for inputs
        input_lower_bound = self._model.inputs["u"].lower_bounds

        for joystick in self.joysticks:
            # XBox 360 Left Stick (left -> right: axis 0, up -> down: axis 1)
            # See https://www.pygame.org/docs/ref/joystick.html#xbox-360-controller-pygame-2-x for more information
            x_force = joystick.get_axis(0) * (input_upper_bound[1] - input_lower_bound[1]) * 0.5
            y_force = joystick.get_axis(1) * (input_upper_bound[0] - input_lower_bound[0]) * 0.5
            input_force = numpy.array([-y_force, -x_force])

        return input_force
