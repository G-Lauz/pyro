import clipy

from pyro.refactor.simulation import PygameSimulation
from pyro.refactor.system import MechanicalSystem, MechanicalSystemConfiguration

from pyro.refactor.boat import (
    # OldBoat2DConfiguration,
    Boat2DConfiguration,
    Boat2DSystemDefinition,
    BoatGeometry,
    # BoatKinematic,
    # BoatParameters,
    DynamicCameraBoatRenderer
)

@clipy.command(usage="python position_control.py --config <path>", description="Boat2D simulation")
@clipy.argument("config", required=True, type=str, help="Path to the configuration file")
def main(config: str):
    """
    Example of usage from root directory:
    ```bash
    python .\examples\refactor\boat\position_control.py --config .\examples\refactor\boat\configuration.yaml
    ```
    """
    # configuration = OldBoat2DConfiguration(config_file=config)
    configuration = Boat2DConfiguration(config_file=config)

    geometry = BoatGeometry(configuration)
    definition = Boat2DSystemDefinition(configuration)
    # kinematic = BoatKinematic(configuration)
    # parameters = BoatParameters(configuration)

    system = MechanicalSystem(geometry=geometry, definition=definition, config=configuration)

    renderer = DynamicCameraBoatRenderer()

    simulation = PygameSimulation(system=system, renderer=renderer)
    simulation.run(render=True)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter