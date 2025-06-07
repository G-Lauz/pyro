import matplotlib.pyplot as plt
import numpy

from scipy.integrate import odeint

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


def plot_signals(history: dict):
    fig, axes = plt.subplots(len(history), 1, figsize=(6, 8))
    for i, (name, values) in enumerate(history.items()):
        axes[i].plot(values)
        axes[i].set_title(f"Signal {name}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel(name)
    plt.tight_layout()


def main():
    """
    Example of usage from root directory:
    ```bash
    python .\examples\refactor\lti\proportional_control.py
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
    lti_system = LTISystem(state_matrix=A, input_matrix=B, output_matrix=C, feedforward_matrix=D, initial_states=numpy.array([1.0]))
    controller = ProportionalController(gain=K, references=ref)

    model = Model(name="LTI System with Controller")
    model.add_system(lti_system)
    model.add_system(controller)

    model.connect(controller, "u", (lti_system, "u"), Signal(name="u",
                                                           dim=1,
                                                           initial_values=0.0,
                                                           lower_bounds=-numpy.inf,
                                                           upper_bounds=numpy.inf))
    model.connect(lti_system, "y", (controller, "obs"), Signal(name="y",
                                                               dim=1,
                                                               initial_values=[1.0],
                                                               lower_bounds=-numpy.inf,
                                                               upper_bounds=numpy.inf))

    model.initialize()

    simulation = ContinuousSimulation(model=lti_system)
    history = simulation.run(dt=0.005, steps=1000, collect=True)

    plot_signals(history)

    dt = 0.05
    time = numpy.arange(0, 5, dt)
    h_time = numpy.arange(0, 5, 0.005)

    def system_only(state, t):
        x = state
        dxdt = A * x
        return dxdt

    x_sol = odeint(system_only, 1.0, time)

    compare_to = history["x"]

    plt.figure(figsize=(8,5))
    plt.plot(time, x_sol[:, 0], label='x(t) from odeint')
    plt.plot(h_time, compare_to, label='x(t) from simulation', alpha=0.75)
    plt.xlabel('Time t')
    plt.ylabel('State x(t)')
    plt.title('Solver vs Simulation (System only)')
    plt.legend()
    plt.grid(True)

    # Reset the simulation and simulate the system
    model.reset()

    simulation = ContinuousSimulation(model=model)
    history = simulation.run(dt=0.005, steps=1000, collect=True)

    plot_signals(history)

    def closed_loop(state, t):
        x = state
        u = -K * (x - ref)
        dxdt = A * x + B * u
        return dxdt

    x_sol = odeint(closed_loop, 1.0, time)
    u = -K * (x_sol[:, 0] - ref)

    compare_to = history["x"]

    plt.figure(figsize=(8,5))
    plt.plot(time, x_sol[:, 0], label='x(t) from odeint')
    plt.plot(h_time, compare_to, label='x(t) from simulation', alpha=0.75)
    plt.xlabel('Time t')
    plt.ylabel('State x(t)')
    plt.title('Solver vs Simulation (Controller + System)')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(8,5))
    plt.plot(time, u, label='u(t) from odeint')
    plt.plot(h_time, history["u"], label='u(t) from simulation', alpha=0.75)
    plt.xlabel('Time t')
    plt.ylabel('Control u(t)')
    plt.title('Solver vs Simulation (Controller + System)')
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
