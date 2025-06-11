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

    def run(self, dt: float = 0.01, fps:int = 60, steps: int = 1000, render:bool = False, collect: bool = False, stop_condition: StopCondition = None):
        """
        Run the simulation using Pygame for rendering and event handling.

        :param dt: The time step for the simulation. (float)
        :param fps: The frames per second for rendering. (int)
        :param steps: The maximum number of simulation steps to run. (int)
        :param render: Whether to render the simulation using Pygame. (bool)
        :param collect: Whether to collect data during the simulation. (bool)
            The data will be collected at each simulation step `dt`.
        :param stop_condition: An optional stop condition to check if the simulation should stop. (StopCondition)

        :return: The collected data if `collect` is True, otherwise None. (dict or None)
        """

        probe = self._create_probe(collect=collect)
        probe.initialize_history(self._model)

        self.clock = self.pygame.time.Clock()
        sim_time = 0.0
        accumulated_sim_time = 0.0
        step_count = 0

        self.is_running = True
        while self.is_running:
            real_dt = self.clock.tick(fps) / 1000.0
            accumulated_sim_time += real_dt

            self._event_handler()

            # Fixed-step simulation
            while accumulated_sim_time >= dt:

                probe.collect_dynamics(self._model, time=sim_time)

                self._model.step(time=sim_time, dt=dt)
                sim_time += dt
                step_count += 1

                probe.collect_statics(self._model, time=sim_time)

                accumulated_sim_time -= dt

                # Stop condition check
                if stop_condition is not None and stop_condition.is_met(self._model):
                    self.is_running = False
                    break

            if render:
                # TODO: won't work with Model, only with System
                # self.renderer.render(self._model.state_signal, self._model.inputs["u"], self._model.outputs["y"])
                # self.renderer.render(self._model.systems["Boat 2D"].state_signal, self._model.systems["Boat 2D"].inputs["u"], self._model.systems["Boat 2D"].outputs["y"])
                self.renderer.render(self._model.systems["Boat 2D"])

        return probe.history if collect else None
