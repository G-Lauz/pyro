from typing import Union

from pyro.refactor.system import System, DynamicSystem
from pyro.refactor.model import Model
from .simulation import Simulation


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
        current_time = 0.0

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

        for _ in range(steps):
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

            current_time += dt

        return history
