import clipy
import numpy

from pyro.refactor.simulation import PygameInteractiveSimulation
from pyro.refactor.system import MechanicalSystem

from pyro.refactor.boat import (
    Boat2DConfiguration,
    BoatGeometry,
    BoatKinematic,
    BoatParameters,
    FixedCameraBoatRenderer,
    DynamicCameraBoatRenderer
)


def trajectory_generator():
    domain = numpy.linspace(0, 30, 100)
    codomain = 10 * numpy.sin(1/16 * numpy.pi * domain) + 5 * numpy.sin(1/8 * numpy.pi * domain)
    trajectory = numpy.array([domain, codomain]).T

    initial_point = trajectory[0]
    trajectory = trajectory - initial_point

    trajectory = trajectory + numpy.array([-4, 0]) # Initial position

    return trajectory


@clipy.command(usage="python boat_sim.py --config <path>", description="Boat2D simulation")
@clipy.argument("config", required=True, type=str, help="Path to the configuration file")
def main(config: str):
    """
    Example of usage from root directory:
    ```bash
    python .\examples\refactor\boat\interactive_simulation.py --config .\examples\refactor\boat\configuration.yaml
    ```
    """
    configuration = Boat2DConfiguration(config_file=config)
    geometry = BoatGeometry(configuration)
    kinematic = BoatKinematic(configuration)
    parameters = BoatParameters(configuration)

    system = MechanicalSystem(parameters=parameters, kinematic=kinematic, geometry=geometry)

    # renderer = FixedCameraBoatRenderer()
    renderer = DynamicCameraBoatRenderer()

    simulation = PygameInteractiveSimulation(system=system, renderer=renderer)
    simulation.run(render=True, trajectory_generator=trajectory_generator)

if __name__ == "__main__":
    main()