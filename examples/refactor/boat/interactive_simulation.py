import clipy

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
    simulation.run(render=True)

if __name__ == "__main__":
    main()