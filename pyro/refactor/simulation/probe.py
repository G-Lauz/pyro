import abc

from typing import Dict, Union

import numpy

from pyro.refactor.model import Model
from pyro.refactor.system import System, DynamicSystem


class Probe(abc.ABC):
    """
    Abstract base class for probes in simulations.
    Probes are used to collect data during the simulation.
    """

    _history: Dict[str, list]

    def __init__(self):
        """
        Initialize the probe.
        This method can be overridden by subclasses to initialize specific attributes.
        """
        self._history = None

    @property
    def history(self) -> Dict[str, numpy.ndarray]:
        # Cast to numpy arrays
        for name, values in self._history.items():
            if isinstance(values, list):
                self._history[name] = numpy.array(values)
        return self._history

    def reset(self):
        """
        Reset the probe's history.
        This method can be called to clear the collected data.
        """
        self._history = {}

    @abc.abstractmethod
    def initialize_history(self, model: Union[Model, System]):
        """
        Initialize the history dictionary for collecting data.
        This method should be called before running the simulation.

        :param model: The model or system to collect data from. (Model or System)
        :return: A dictionary with signal names as keys and empty lists as values.
        """

    @abc.abstractmethod
    def collect_dynamics(self, model: Union[Model, System], time: float):
        """
        Collect dynamics data from the model or system at a specific time.

        :param model: The model or system to collect data from. (Model or System)
        :param time: The current simulation time.              (float)
        """

    @abc.abstractmethod
    def collect_statics(self, model: Union[Model, System], time: float):
        """
        Collect static data from the model or system at a specific time.

        :param model: The model or system to collect data from. (Model or System)
        :param time: The current simulation time.              (float)
        """


class NullProbe(Probe):
    """
    A null probe that does not collect any data.
    This is useful when no data collection is needed.
    """
    def initialize_history(self, model: Union[Model, System]):
        """
        Initialize the history for a null probe.
        This method does nothing as no data is collected.
        """
        pass

    def collect_dynamics(self, model: Union[Model, System], time: float):
        pass

    def collect_statics(self, model: Union[Model, System], time: float):
        pass


class ModelProbe(Probe):
    """
    A probe that collects data from a model.
    It collects both dynamics and static data.
    """
    def initialize_history(self, model: Model):
        signals = {}
        for system in model.systems.values():
            # Collect all signals
            for signal in system.outputs.values():
                signals[signal.name] = signal

            # Collect all internal states
            if isinstance(system, DynamicSystem):
                signals[system.state_signal.name] = system.states

        self._history = {name: [] for name in signals.keys()}

    def collect_dynamics(self, model: Model, time: float):
        if not self._history:
            raise ValueError("History is not initialized. Call initialize_history() first.")

        for system in model.systems.values():
            if isinstance(system, DynamicSystem):
                for name, signal in system.outputs.items():
                    signal_sample = numpy.stack([time * numpy.ones_like(signal.values), signal.values.copy()], axis=-1)
                    self._history[name].append(signal_sample)
                state_sample = numpy.stack([time * numpy.ones_like(system.states), system.states.copy()], axis=-1)
                self._history[system.state_signal.name].append(state_sample)

    def collect_statics(self, model: Model, time: float):
        if not self._history:
            raise ValueError("History is not initialized. Call initialize_history() first.")

        for system in model.systems.values():
            if not isinstance(system, DynamicSystem):
                for name, signal in system.outputs.items():
                    signal_sample = numpy.stack([time * numpy.ones_like(signal.values), signal.values.copy()], axis=-1)
                    self._history[name].append(signal_sample)


class SystemProbe(Probe):
    """
    A probe that collects data from a system.
    It collects both dynamics and static data.
    """

    def initialize_history(self, model: System):
        signals = {}

        # Collect all signals
        for signal in model.outputs.values():
            signals[signal.name] = signal

        # Collect all internal states
        if isinstance(model, DynamicSystem):
            signals[model.state_signal.name] = model.states

        self._history = {name: [] for name in signals.keys()}

    def collect_dynamics(self, model: System, time: float):
        if not self._history:
            raise ValueError("History is not initialized. Call initialize_history() first.")

        if isinstance(model, DynamicSystem):
            for name, signal in model.outputs.items():
                signal_sample = numpy.stack([time * numpy.ones_like(signal.values), signal.values.copy()], axis=-1)
                self._history[name].append(signal_sample)
            state_sample = numpy.stack([time * numpy.ones_like(model.states), model.states.copy()], axis=-1)
            self._history[model.state_signal.name].append(state_sample)

    def collect_statics(self, model: System, time: float):
        if not self._history:
            raise ValueError("History is not initialized. Call initialize_history() first.")

        if not isinstance(model, DynamicSystem):
            for name, signal in model.outputs.items():
                signal_sample = numpy.stack([time * numpy.ones_like(signal.values), signal.values.copy()], axis=-1)
                self._history[name].append(signal_sample)
