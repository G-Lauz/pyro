import matplotlib.pyplot as plt
import numpy

from pyro.refactor.signal import Signal
from pyro.refactor.system import StaticSystem
from pyro.refactor.dynamics import LTISystem
from pyro.refactor.model import Model
from pyro.refactor.simulation import ContinuousSimulation


class ProportionalController(StaticSystem):
    """
    Proportional controller for LTI systems.
    """

    def __init__(self, gain: float, references: float = None):
        """
        Initialize the proportional controller.

        :param gain: The proportional gain of the controller. (float)
        """
        super().__init__(name="Proportional Controller")

        self.add_input_port(name="obs")
        self.add_output_port(name="u")

        self.gain = -gain
        self.references = references

    def compute_output(self, time: float, dt: float = 0.01) -> numpy.ndarray:
        """
        Compute the output of the controller.

        :param time: The current time of the simulation. (float)
        :param dt: The time step of the simulation. (float)
        :return: The output of the controller. (numpy.ndarray)
        """
        if self.references is None:
            raise ValueError("References signal is not set.")

        # Compute the error
        error = self.inputs["obs"].values - self.references

        output = self.gain * error
        return {
            "u": output
        }


class ConstantSignal(StaticSystem):
    """
    Constant signal generator.
    """

    def __init__(self, name: str, value: float):
        """
        Initialize the constant signal generator.

        :param value: The constant value of the signal. (float)
        """
        super().__init__(name=name)

        self.add_output_port(name="y")

        self.value = value

    def compute_output(self, time: float, dt: float = 0.01) -> numpy.ndarray:
        """
        Compute the output of the constant signal generator.

        :param time: The current time of the simulation. (float)
        :param dt: The time step of the simulation. (float)
        :return: The output of the constant signal generator. (numpy.ndarray)
        """
        return numpy.full(self.outputs["y"].dim, self.value)


def plot_signals(history: dict):
    fig, axes = plt.subplots(len(history), 1, figsize=(6, 8))
    for i, (name, values) in enumerate(history.items()):
        axes[i].plot(values)
        axes[i].set_title(f"Signal {name}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel(name)
    plt.tight_layout()
    plt.show()


def main():
    """
    Example of usage from root directory:
    ```bash
    python .\examples\refactor\lti\simulation.py
    ```
    """
    # Define the state-space matrices
    A = -1.0
    B = 1.0
    C = 1.0
    D = 0.0

    K = 2.0
    ref = 5.0

    # Create an LTI system
    lti_system = LTISystem(state_matrix=A, input_matrix=B, output_matrix=C, feedforward_matrix=D)
    controller = ProportionalController(gain=K, references=ref)

    model = Model(name="LTI System with Controller")
    model.add_system(lti_system)
    model.add_system(controller)

    model.connect(controller, "u", (lti_system, "u"), Signal(name="u",
                                                           dim=1,
                                                           initial_values=numpy.zeros(1),
                                                           lower_bounds=numpy.full(1, -numpy.inf),
                                                           upper_bounds=numpy.full(1, numpy.inf)))
    model.connect(lti_system, "y", (controller, "obs"), Signal(name="y",
                                                               dim=1,
                                                               initial_values=numpy.zeros(1),
                                                               lower_bounds=numpy.full(1, -numpy.inf),
                                                               upper_bounds=numpy.full(1, numpy.inf)))
    
    model.initialize()

    # Set the initial states
    lti_system.states.update(numpy.array([0.0]))

    # Simulate the lti system only
    lti_system.states.update(numpy.array([1.0]))

    simulation = ContinuousSimulation(model=lti_system)
    history = simulation.run(dt=0.05, steps=100, collect=True)

    plot_signals(history)

    # Simulate the system
    # Reset the simulation
    lti_system.states.update(numpy.array([1.0]))
    lti_system.inputs["u"].update(numpy.array([0.0]))
    lti_system.outputs["y"].update(numpy.array([1.0]))

    simulation = ContinuousSimulation(model=model)
    history = simulation.run(dt=0.05, steps=100, collect=True)

    plot_signals(history)

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
