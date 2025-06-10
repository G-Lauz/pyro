from typing import Union

from pyro.refactor.system import System, DynamicSystem
from pyro.refactor.model import Model

from .probe import Probe, NullProbe, ModelProbe, SystemProbe
from .simulation import Simulation
from .stop_condition import StopCondition


class ContinuousSimulation(Simulation):
    def __init__(self, model: Union[Model, System]): # TODO: make a common interface for model and system
        """
        Initialize the continuous simulation with a model or system.

        :param model: The model or system to simulate.              (Model or System)
        """
        self._model = model

    def run(self, dt: float = 0.01, steps: int = 1000, collect: bool = False, stop_condition: StopCondition = None):
        """
        Run the simulation for a given time and time step.

        :param dt: The time step of the simulation.                         (float)
        :param steps: The number of steps to run.                           (int)
        :param collect: Whether to collect history the simulation.          (bool)
        """
        current_time = 0.0

        probe = self._create_probe(collect=collect)
        probe.initialize_history(self._model)

        for _ in range(steps):
            probe.collect_dynamics(self._model, time=current_time)
            self._model.step(time=current_time, dt=dt)
            probe.collect_statics(self._model, time=current_time)

            current_time += dt

            if stop_condition is not None and stop_condition.is_met(self._model):
                break

        return probe.history if collect else None

    def _create_probe(self, collect=False):
        """
        Create a probe for the simulation.

        :param collect: Whether to collect history the simulation. (bool)
        :return: A probe instance for the simulation.
        """
        if collect:
            if isinstance(self._model, Model):
                return ModelProbe()

            elif isinstance(self._model, System):
                return SystemProbe()

        return NullProbe()
