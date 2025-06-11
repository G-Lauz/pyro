import clipy
import matplotlib.pyplot as plt
import numpy

from scipy.integrate import solve_ivp

from pyro.refactor.simulation import PygameSimulation
from pyro.refactor.dynamics.mechanical.system import MechanicalSystem
from pyro.refactor.signal import Signal
from pyro.refactor.dynamics.mechanical.signal import MechanicalStateSignal
from pyro.refactor.system import StaticSystem, DynamicSystem
from pyro.refactor.model import Model

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


class PIDController(StaticSystem):
    """
    Proportional-Derivative controller.
    """

    def __init__(self, kp: float, kd:float, ki: float, references: Signal = None):
        """
        Initialize the proportional controller.

        :param gain: The proportional gain of the controller. (float)
        :param references: The reference signal to track. (Signal)
        """
        super().__init__(name="Proportional Controller")

        self.add_input_port(name="obs")
        self.add_output_port(name="u")

        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.references = references

        self.previous_error = None

    def compute_output(self, time: float, dt: float = 0.01) -> dict:
        obs = self.inputs["obs"].values
        theta = obs[2]

        # Error computation
        if self.references is None:
            ref = numpy.zeros(6)
        else:
            ref = self.references

        error = ref[:3] - obs[:3]
        self.previous_error = error

        derror = (error - self.previous_error) / dt

        # Integral term computation
        if not hasattr(self, '_integral_error'):
            self._integral_error = numpy.zeros(3)
        self._integral_error += error * dt

        # PID control law
        u = (self.kp * error +
             self.kd * derror + 
             self.ki * self._integral_error)

        # Convert to force vector
        Fx = u[0] * numpy.cos(theta) + u[1] * numpy.sin(theta)
        Fy = u[0] * numpy.sin(theta) - u[1] * numpy.cos(theta)

        return {"u": numpy.array([Fx, Fy])}
    

def plot_signals(history: dict, title: str = "Boat 2D Signals"):
    fig, axes = plt.subplots(len(history), 1, figsize=(6, 8), squeeze=False)
    axes = axes.flatten()

    fig.suptitle(title)

    for i, (name, values) in enumerate(history.items()):
        axes[i].plot(values[:,:,0], values[:,:,1])
        axes[i].set_title(f"Signal {name}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel(name)

    plt.tight_layout()


@clipy.command(usage="python position_control.py --config <path>", description="Boat2D simulation")
# @clipy.argument("config", required=True, type=str, help="Path to the configuration file")
# def main(config: str):
def main():
    """
    Example of usage from root directory:
    ```bash
    python ./examples/refactor/boat/proportional_control.py
    ```
    """
    geometry = BoatGeometry(CONFIGURATION.geometry)
    kinematics = BoatKinematics()
    dynamics = BoatDynamics(CONFIGURATION)

    system = MechanicalSystem(name="Boat 2D", kinematics=kinematics, dynamics=dynamics)
    # TODO: Allow system only simulation without state signal definition
    system.state_signal = MechanicalStateSignal(name="x",
                                                dim=6,
                                                initial_values= [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                lower_bounds=[-10.0, -10.0, -numpy.pi, -10.0, -10.0, -numpy.pi],
                                                upper_bounds=[10.0, 10.0, numpy.pi, 10.0, 10.0, numpy.pi])

    controller = PIDController(kp=300, kd=0, ki=0, references=[-5.0, -5.0, 0.0, 0.0, 0.0, 0.0])

    model = Model(name="Boat 2D with proportional controller")
    model.add_system(system)
    model.add_system(controller)

    model.connect(controller, "u", (system, "u"), Signal(name="u",
                                                         dim=2,
                                                         initial_values=[0.0, 0.0],
                                                         lower_bounds=[-10000.0, -1000.0],
                                                         upper_bounds=[10000.0, 1000.0],
                                                         saturation=True))
    model.connect(system, "y", (controller, "obs"), Signal(name="y",
                                                         dim=6,
                                                         initial_values=[5.0, 5.0, 0.0, 0.0, 0.0, 0.0],
                                                         lower_bounds=[-10.0, -10.0, -numpy.pi, -10.0, -10.0, -numpy.pi],
                                                         upper_bounds=[10.0, 10.0, numpy.pi, 10.0, 10.0, numpy.pi]))

    model.initialize()

    # Solve using solve_ivp
    solution = solve_ivp(
        fun=lambda t, y: model.dynamics(t, y, dt=0.01),
        t_span=(0, 100),
        y0=model.get_initial_states(),
        t_eval=numpy.arange(0, 100, 0.01),
        method='RK45'
    )

    # Create a history dictionary to store signals
    history = {}
    for system in model.ordered_systems:
        if isinstance(system, DynamicSystem):
            t_repeated = numpy.tile(solution.t, (6, 1)).T  # Shape becomes (10000, 6)
            history[system.state_signal.name] = numpy.stack((t_repeated, solution.y[:len(system.state_signal.values), :].T), axis=-1)

    plot_signals(history, title="Signals from solve_ivp")

    model.reset()

    renderer = BoatRenderer(geometry=geometry)

    simulation = PygameSimulation(model=model, renderer=renderer)
    history = simulation.run(render=True, collect=True)

    plot_signals(history, title="Signals from Pygame Simulation")
    plt.show()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
