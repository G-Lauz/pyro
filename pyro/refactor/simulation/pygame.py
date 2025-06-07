from typing import Union

import numpy

from pyro.refactor.system import System, DynamicSystem
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

    def run(self, dt=0.1, steps=1000, render=False, callback=None, collect=False):
        self.is_running = True

        current_time = 0.0
        self.clock = self.pygame.time.Clock()
        dt = self.clock.tick(60) / 1000

        history = None

        if collect:
            signals = {}

            if isinstance(self._model, System):
                # Collect all signals
                for signal in self._model.outputs.values():
                    signals[signal.name] = signal

                # Collect all internal states
                if isinstance(self._model, DynamicSystem):
                    signals[self._model.state_signal.name] = self._model.states

            elif isinstance(self._model, Model):
                for system in self._model.systems.values():
                    # Collect all signals
                    for signal in system.outputs.values():
                        signals[signal.name] = signal

                    # Collect all internal states
                    if isinstance(system, DynamicSystem):
                        signals[system.state_signal.name] = system.states

            history = {name: [] for name in signals.keys()}

        while self.is_running:
            self._event_handler()

            if isinstance(self._model, Model):
                # Collect signals from dynamic systems
                if collect:
                    for system in self._model.systems.values():
                        if isinstance(system, DynamicSystem):
                            for name, signal in system.outputs.items():
                                history[name].append(signal.values.copy())
                            history[system.state_signal.name].append(system.states.copy())

                self._model.step(time=current_time, dt=dt)

                # Collect signals from static systems
                if collect:
                    for system in self._model.systems.values():
                        if not isinstance(system, DynamicSystem):
                            for name, signal in system.outputs.items():
                                history[name].append(signal.values.copy())

            elif isinstance(self._model, System):
                # Collect signals from dynamic systems
                if collect and isinstance(self._model, DynamicSystem):
                    for name, signal in self._model.outputs.items():
                        history[name].append(signal.values.copy())
                    history[self._model.state_signal.name].append(self._model.states.copy())

                self._model.step(time=current_time, dt=dt)

                # Collect signals from static systems
                if collect and not isinstance(self._model, DynamicSystem):
                    for name, signal in self._model.outputs.items():
                        history[name].append(signal.values.copy())

            if callback is not None:
                raise NotImplementedError("Callback is not implemented in PygameSimulation")
                callback()

            if render:
                # TODO: won't work with Model, only with System
                # self.renderer.render(self._model.state_signal, self._model.inputs["u"], self._model.outputs["y"])
                # self.renderer.render(self._model.systems["Boat 2D"].state_signal, self._model.systems["Boat 2D"].inputs["u"], self._model.systems["Boat 2D"].outputs["y"])
                self.renderer.render(self._model.systems["Boat 2D"])

            dt = self.clock.tick(60) / 1000
            current_time += dt

        return history


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
                # self.renderer.render(self._model.state_signal, self._model.inputs["u"], self._model.outputs["y"])
                self.renderer.render(self._model)

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
