import abc

from typing import Union

from pyro.refactor.system import System, DynamicSystem
from pyro.refactor.model import Model

class Simulation(abc.ABC):
    @abc.abstractmethod
    def run(self, dt: float = 0.01, steps: int = 1000):
        """
        Run the simulation for a given time and time step.

        :param time: The current time of the simulation.            (float)
        :param dt: The time step of the simulation.                 (float)
        """


class ContinuousSimulation(Simulation):
    def __init__(self, model: Union[Model, System]): # TODO: make a common interface for model and system
        """
        Initialize the continuous simulation with a model or system.

        :param model: The model or system to simulate.              (Model or System)
        """
        self._model = model

    def run(self, dt: float = 0.01, steps: int = 1000, collect: bool = False):
        """
        Run the simulation for a given time and time step.

        :param dt: The time step of the simulation.                         (float)
        :param steps: The number of steps to run.                           (int)
        :param collect: Whether to collect history the simulation.          (bool)
        """
        history = None

        if collect:
            signals = {}

            if isinstance(self._model, System):
                # Collect all signals
                for signal in self._model.outputs.values():
                    signals[signal.name] = signal

                # Collect all internal states
                if isinstance(self._model, DynamicSystem):
                    signals[self._model.states.name] = self._model.states

            elif isinstance(self._model, Model):
                for system in self._model.systems.values():
                    # Collect all signals
                    for signal in system.outputs.values():
                        signals[signal.name] = signal

                    # Collect all internal states
                    if isinstance(system, DynamicSystem):
                        signals[system.states.name] = system.states

            history = {name: [] for name in signals.keys()}

        for _ in range(steps):
            sig = self._model.step(dt=dt)
            self._model.update(sig)

            if collect:
                for name, signal in signals.items():
                    history[name].append(signal.values.copy())

        return history
