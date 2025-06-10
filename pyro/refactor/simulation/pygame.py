from typing import Union

from pyro.refactor.system import System
from pyro.refactor.model import Model
from pyro.refactor.renderer import Renderer

from .continuous import ContinuousSimulation
from .stop_condition import StopCondition


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

    def run(self, dt=0.1, steps=1000, render=False, callback=None, collect=False, stop_condition: StopCondition = None):
        self.is_running = True

        current_time = 0.0
        self.clock = self.pygame.time.Clock()
        dt = self.clock.tick(60) / 1000

        probe = self._create_probe(collect=collect)
        probe.initialize_history(self._model)

        while self.is_running:
            self._event_handler()

            probe.collect_dynamics(self._model, time=current_time)
            self._model.step(time=current_time, dt=dt)
            probe.collect_statics(self._model, time=current_time)

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

            if stop_condition is not None and stop_condition.is_met(self._model):
                self.is_running = False

        return probe.history if collect else None
