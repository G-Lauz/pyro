import abc

from typing import Dict, Optional

import numpy

from pyro.refactor.signal import Signal, StateSignal


class System(abc.ABC):
    name: str

    inputs: Dict[str, Optional[Signal]]
    outputs: Dict[str, Optional[Signal]]

    def __init__(self, name: str):
        """
        Initialize the system with state, input and output signals.

        :param name: The name of the system.                        (str)
        :param states: The state signal of the system.              (Signal)
        :param inputs: The input signal of the system.              (Signal)
        :param outputs: The output signal of the system.            (Signal)
        :param signals: Additional signals of the system.           (dict)
        """
        self.name = name
        self.inputs = {}
        self.outputs = {}

    @abc.abstractmethod
    def step(self, time: float = 1.0, dt: float = 0.01) -> Dict[str, numpy.ndarray]:
        """
        Step the system forward in time.

        :param time: The current time of the simulation.            (float)
        :param dt: The time step of the simulation.                 (float)
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Reset the system to its initial state.

        This method should reset all internal states and signals of the system.
        """

    def update(self, signals: Dict[str, numpy.ndarray]) -> None:
        """
        Update the system with new signals.

        :param signals: The signals to update the system with.      (dict)
        """
        for name, signal in signals.items():
            if name in self.outputs:
                self.outputs[name].update(signal)
            else:
                raise ValueError(f"Signal {name} not found in outputs of system {self.name}.")

    @abc.abstractmethod
    def compute_output(self, time: float, dt: float = 0.01) -> Dict[str, numpy.ndarray]:
        """
        Compute the output of the system from the state and the control input.

        \equation{begin}
            y = h(x, u, t)
        \equation{end}

        :param time: The current time of the simulation.            (float)
        :param dt: The time step of the simulation.                 (float)

        :return: The output vector.                                 (output_dim, 1)
        """

    def add_input_port(self, name: str):
        """
        Add an input signal to the block.

        :param signal: The input signal to add.                     (Signal)
        """
        self.inputs[name] = None

    def add_output_port(self, name: str):
        """
        Add an output signal to the block.

        :param signal: The output signal to add.                    (Signal)
        """
        self.outputs[name] = None


class StaticSystem(System):
    def __init__(self, name: str):
        super().__init__(name=name)

    def step(self, time: float = 1.0, dt: float = 0.01) -> Dict[str, numpy.ndarray]:
        """
        Step the system forward in time.

        :param time: The current time of the simulation.            (float)
        :param dt: The time step of the simulation.                 (float)
        """
        # No dynamics to compute for a static block since there are no states
        output_signals = self.compute_output(time=time, dt=dt)
        self.update(output_signals)

    def reset(self) -> None:
        """
        Reset the system to its initial state.

        This method should reset all internal states and signals of the system.
        """
        for signal in self.outputs.values():
            if signal is not None:
                signal.reset()


class DynamicSystem(StaticSystem):
    states: StateSignal

    def __init__(self, name: str, state_signal: StateSignal):
        super().__init__(name=name)

        self.state_signal = state_signal

    @property
    def states(self) -> numpy.ndarray:
        """
        The state signal of the system.                            (Signal)
        """
        return self.state_signal.states

    @states.setter
    def states(self, values: numpy.ndarray):
        """
        Set the state signal of the system.
        :param values: The state signal of the system.              (numpy.ndarray)
        """
        self.state_signal.states = values

    def step(self, time: float = 1.0, dt: float = 0.01) -> Dict[str, numpy.ndarray]:
        """
        Step the system forward in time.

        :param time: The current time of the simulation.            (float)
        :param dt: The time step of the simulation.                 (float)
        """
        dynamics = self.compute_dynamics(time=time, dt=dt)
        self.states = self.states + dynamics * dt

        output_signals = self.compute_output(time=time, dt=dt)
        self.update(output_signals)

    @abc.abstractmethod
    def compute_dynamics(self, time: float, dt: float = 0.01) -> numpy.ndarray:
        """
        Compute the dynamics of the system from the state, the control input and the time step.

        \equation{begin}
            dx = f(x, u, t)
        \equation{end}

        :param time: The current time of the simulation.            (float)
        :param dt: The time step of the simulation.                 (float)

        :return: The state derivative vector.                       (state_dim, 1)
        """

    def reset(self) -> None:
        """
        Reset the system to its initial state.

        This method should reset all internal states and signals of the system.
        """
        super().reset()
        if self.states is not None:
            self.state_signal.reset()
