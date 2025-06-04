import clipy
import numpy

from pyro.refactor.simulation import PygameSimulation, PygameInteractiveSimulation
from pyro.refactor.dynamics.mechanical.system import MechanicalSystem
from pyro.refactor.signal import Signal
from pyro.refactor.dynamics.mechanical.signal import MechanicalStateSignal

from pyro.refactor.dynamics.mechanical.boat import (
    BoatConfiguration,
    BoatDynamics,
    BoatKinematics,
    BoatGeometry,
    DynamicCameraBoatRenderer
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


@clipy.command(usage="python position_control.py --config <path>", description="Boat2D simulation")
# @clipy.argument("config", required=True, type=str, help="Path to the configuration file")
# def main(config: str):
def main():
    """
    Example of usage from root directory:
    ```bash
    python .\examples\refactor\boat\simulation.py
    ```
    """
    geometry = BoatGeometry(CONFIGURATION.geometry)
    kinematics = BoatKinematics()
    dynamics = BoatDynamics(CONFIGURATION)

    system = MechanicalSystem(name="Boat 2D", kinematics=kinematics, dynamics=dynamics)
    system.add_input_port(name="u")
    system.add_output_port(name="y")

    # TODO: Allow system only simulation without inputs and outputs signal deifnition
    system.inputs["u"] = Signal(name="u",
                                dim=2,
                                initial_values=[0.0, 0.0],
                                lower_bounds=[-10000.0, -1000.0],
                                upper_bounds=[10000.0, 1000.0])
    system.outputs["y"] = Signal(name="y",
                                 dim=6,
                                 initial_values=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                 lower_bounds=[-10.0, -10.0, -numpy.pi, -10.0, -10.0, -numpy.pi],
                                 upper_bounds=[10.0, 10.0, numpy.pi, 10.0, 10.0, numpy.pi])
    # TODO: Allow system only simulation without state signal definition
    system.state_signal = MechanicalStateSignal(name="x",
                                                dim=6,
                                                initial_values= numpy.zeros(6),
                                                lower_bounds=[-10.0, -10.0, -numpy.pi, -10.0, -10.0, -numpy.pi],
                                                upper_bounds=[10.0, 10.0, numpy.pi, 10.0, 10.0, numpy.pi])

    renderer = DynamicCameraBoatRenderer(geometry=geometry)

    # simulation = PygameSimulation(model=system, renderer=renderer)
    simulation = PygameInteractiveSimulation(system=system, renderer=renderer)
    simulation.run(render=True)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter