# Patch to load module
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
print(os.path.join(os.path.dirname(__file__), '..'))

import clipy
import numpy
import pygame

from refactor.simulation import ContinuousSimulation
from refactor.system import MechanicalSystem

from refactor.boat import (
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
    # Boat definition
    configuration = Boat2DConfiguration(config_file=config)
    geometry = BoatGeometry(configuration)
    kinematic = BoatKinematic(configuration)
    parameters = BoatParameters(configuration)

    system = MechanicalSystem(parameters=parameters, kinematic=kinematic)

    simulation = ContinuousSimulation(system=system)

    # renderer = FixedCameraBoatRenderer()
    renderer = DynamicCameraBoatRenderer()

    # Simulation
    running = True
    clock = pygame.time.Clock()
    pygame.joystick.init()
    joysticks = []

    dt = clock.tick(60) / 1000

    # Sinusoidal 2D trajectory
    domain = numpy.linspace(0, 30, 100)
    codomain = 10 * numpy.sin(1/16 * numpy.pi * domain) + 5 * numpy.sin(1/8 * numpy.pi * domain)
    trajectory = numpy.array([domain, codomain]).T

    while running:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.JOYDEVICEADDED:
                joystick = pygame.joystick.Joystick(event.device_index)
                joysticks.append(joystick)

        for joystick in joysticks:
            # XBox 360 Left Stick (left -> right: axis 0, up -> down: axis 1)
            # See https://www.pygame.org/docs/ref/joystick.html#xbox-360-controller-pygame-2-x for more information
            x_force = joystick.get_axis(0) * (parameters.inputs_upper_bound[1] - parameters.inputs_lower_bound[1]) * 0.5
            y_force = joystick.get_axis(1) * (parameters.inputs_upper_bound[0] - parameters.inputs_lower_bound[0]) * 0.5
            input_force = numpy.array([-y_force, -x_force])

        simulation.step(input_force, dt)
        # renderer.render(geometry, next_states, input_force=input)
        renderer.render(geometry, simulation._system.kinematic, input_force=input_force, trajectory=trajectory)

        dt = clock.tick(60) / 1000


if __name__ == "__main__":
    main()
