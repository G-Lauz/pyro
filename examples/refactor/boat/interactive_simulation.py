import clipy
import matplotlib.pyplot as plt
import numpy

from pyro.refactor.simulation import PygameSimulation
from pyro.refactor.dynamics.mechanical.system import MechanicalSystem
from pyro.refactor.signal import Signal, StateSignal
from pyro.refactor.dynamics.mechanical.signal import MechanicalStateSignal
from pyro.refactor.controller import PygameJoystickController
from pyro.refactor.model import Model
from pyro.refactor.system import StaticSystem, DynamicSystem

from pyro.refactor.dynamics.mechanical.boat import (
    BoatConfiguration,
    BoatDynamics,
    BoatKinematics,
    BoatGeometry,
    BoatRenderer
)

from pyro.refactor.dynamics.mechanical.configuration import MechanicalConfiguration
from pyro.refactor.dynamics.mechanical.boat.configuration import (
    BoatGeometryConfiguration,
    HydrodynamicsConfiguration,
    QuadraticDampingConfiguration
)


CONFIGURATION = BoatConfiguration(
    mechanical=MechanicalConfiguration(
        dof=3, # x, y, theta
        mass=1000.0,
        inertia=1000.0,
    ),
    hydrodynamics=HydrodynamicsConfiguration(
        water_density=1000.0,
        linear_damping=[2000.0, 20000.0, 10000.0],
        quadratic_damping=QuadraticDampingConfiguration(
            cx_max=0.5,
            cy_max=0.6,
            cm_max=0.1
        )
    ),
    geometry=BoatGeometryConfiguration(
        thrust_offset=3.0,  # Offset of the thrust force from the center of mass
        lateral_area=6.0,   # Lateral area for hydrodynamic forces
        frontal_area=1.5,   # Frontal area for hydrodynamic forces
        length_overall=6.0  # Length of the boat
    )
)

class PIDController(DynamicSystem):
    def __init__(self):
        """
        Initialize the proportional controller.

        :param gain: The proportional gain of the controller. (float)
        :param references: The reference signal to track. (Signal)
        """
        state_signal = StateSignal(name="pid_state",
                                   dim=9, # (error, derror, integral_error)
                                   initial_values=numpy.zeros(9),
                                   lower_bounds=numpy.full(9, -numpy.inf),
                                   upper_bounds=numpy.full(9, numpy.inf))

        super().__init__(name="Proportional Controller", state_signal=state_signal)

        self.add_input_port(name="obs")
        self.add_input_port(name="ref")
        self.add_output_port(name="v")

        self.kp = numpy.array([5.0, 0.0, 2.0])  # Proportional gain for each state
        self.kd = numpy.array([0.0, 0.0, 0.0])  # Derivative gain for each state
        self.ki = numpy.array([0.0, 0.0, 0.0])  # Integral gain for each state

    def compute_output(self, time: float, dt: float = 0.01) -> dict:
        error = self.states[:3]  # Error
        derror = self.states[3:6]  # Derivative of error
        integral_error = self.states[6:]  # Integral of error

        # PID control law
        u = (self.kp * error +
             self.kd * derror +
             self.ki * integral_error)

        # From (surge rate, sway rate, yaw rate) to (Fx, Fy)
        u = numpy.array([u[0], -u[2]])

        u = numpy.clip(u, self.outputs["v"].lower_bounds, self.outputs["v"].upper_bounds)

        return {"v": u}

    def compute_dynamics(self, time, dt = 0.01):
        obs = self.inputs["obs"].values
        ref = self.inputs["ref"].values

        previous_error = self.states[:3]
        integral_error = self.states[6:]

        target_velocity = numpy.array([ref[0], 0.0, ref[1]])
        system_velocity = obs[3:]

        # Proportional term computation
        error = target_velocity - system_velocity

        # Derivative term computation
        derror = (error - previous_error) / dt**2

        # Integral term computation
        integral_error_derivative = error
        error_derivative = error / dt

        return numpy.concatenate(([error_derivative, derror, integral_error_derivative]), axis=0)
        # self.states = state

        # return 0.0 # no update needed for PID controller


def plot_signals(history: dict, title: str = "Boat 2D Signals"):
    # fig, axes = plt.subplots(len(history), 1, figsize=(6, 8), squeeze=False)
    # axes = axes.flatten()

    # fig.suptitle(title)

    # for i, (name, values) in enumerate(history.items()):
    #     axes[i].plot(values[:,:,0], values[:,:,1])
    #     axes[i].set_title(f"Signal {name}")
    #     axes[i].set_xlabel("Time")
    #     axes[i].set_ylabel(name)

    # plt.tight_layout()

    # keep first 3 dimension of pid state
    history["pid_state"] = history["pid_state"][:, :3, :]

    label_map = {
        "x": ["x", "y", "theta", "dx", "dy", "dtheta"],
        "u": ["Fx", "Fy"],
        "y": ["x", "y", "theta", "dx", "dy", "dtheta"],
        "v": ["Fx", "Fy"],
        "pid_state": ["error_x", "error_y", "error_theta"]
    }

    for i, (name, values) in enumerate(history.items()):
        fig, axes = plt.subplots(values.shape[1], 1, figsize=(6, 8), squeeze=False)
        axes = axes.flatten()

        fig.suptitle(f"{title} - {name}")

        for j in range(values.shape[1]):
            axes[j].plot(values[:, j, 0], values[:, j, 1])
            axes[j].set_xlabel("Time")
            axes[j].set_ylabel(label_map.get(name, [name])[j])
            axes[j].grid(True)

        plt.tight_layout()


@clipy.command(usage="python position_control.py --config <path>", description="Boat2D simulation")
# @clipy.argument("config", required=True, type=str, help="Path to the configuration file")
# def main(config: str):
def main():
    """
    Example of usage from root directory:
    ```bash
    python ./examples/refactor/boat/interactive_simulation.py
    ```
    """
    geometry = BoatGeometry(CONFIGURATION.geometry)
    kinematics = BoatKinematics()
    dynamics = BoatDynamics(CONFIGURATION)

    system = MechanicalSystem(name="Boat 2D", kinematics=kinematics, dynamics=dynamics)
    # TODO: Allow system only simulation without state signal definition
    system.state_signal = MechanicalStateSignal(name="x",
                                                dim=6,
                                                initial_values= numpy.zeros(6),
                                                lower_bounds=[-10.0, -10.0, -numpy.pi, -10.0, -10.0, -numpy.pi],
                                                upper_bounds=[10.0, 10.0, numpy.pi, 10.0, 10.0, numpy.pi])

    # TODO: Allow system only simulation without inputs and outputs signal deifnition
    system.add_output_port(name="y")
    system.outputs["y"] = Signal(name="y",
                                 dim=6,
                                 initial_values=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                 lower_bounds=[-10.0, -10.0, -numpy.pi, -10.0, -10.0, -numpy.pi],
                                 upper_bounds=[10.0, 10.0, numpy.pi, 10.0, 10.0, numpy.pi])
    
    low_level_controller = PIDController()
    controller = PygameJoystickController()

    model = Model(name="Boat 2D with joystick controller")
    model.add_system(system)
    model.add_system(low_level_controller)
    model.add_system(controller)
    model.connect(controller, "u", (low_level_controller, "ref"), Signal(name="u",
                                                                         dim=2,
                                                                         initial_values=[0.0, 0.0],
                                                                         lower_bounds=[-8.0, -2.0],
                                                                         upper_bounds=[8.0, 2.0],
                                                                         saturation=True))
    model.connect(system, "y", (low_level_controller, "obs"), Signal(name="y",
                                                                     dim=6,
                                                                     initial_values=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                                     lower_bounds=[-10.0, -10.0, -numpy.pi, -10.0, -10.0, -numpy.pi],
                                                                     upper_bounds=[10.0, 10.0, numpy.pi, 10.0, 10.0, numpy.pi]))
    model.connect(low_level_controller, "v", (system, "u"), Signal(name="v",
                                                                     dim=2,
                                                                     initial_values=[0.0, 0.0],
                                                                     lower_bounds=[-4000.0, -1500.0],
                                                                     upper_bounds=[4000.0, 1500.0],
                                                                     saturation=True))
    model.initialize()

    renderer = BoatRenderer(geometry=geometry)

    simulation = PygameSimulation(model=model, renderer=renderer)
    history = simulation.run(dt=0.01, render=True, collect=True)

    plot_signals(history, title="Signals from Pygame Simulation")
    plt.show()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter