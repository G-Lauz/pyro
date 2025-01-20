# Patch to load module
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy
import pygame

from refactor.simulation import ContinuousSimulation
from refactor.system import MechanicalSystem
from refactor.renderer import PygameRenderer, DynamicCameraBoatRenderer

from refactor.boat import BoatGeometry, BoatKinematic, BoatParameters


def main():
    # Boat definition
    geometry = BoatGeometry()
    kinematic = BoatKinematic()
    parameters = BoatParameters()

    system = MechanicalSystem(parameters=parameters, kinematic=kinematic)

    simulation = ContinuousSimulation(system=system)

    # renderer = PygameRenderer()
    renderer = DynamicCameraBoatRenderer()

    # Simulation
    running = True
    clock = pygame.time.Clock()
    pygame.joystick.init()
    joysticks = []

    dt = clock.tick(60) / 1000
    next_states = numpy.array([0, 0, 0, 0, 0, 0])

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
            input = numpy.array([-y_force, -x_force])

        next_states = simulation.step(next_states, input, dt)
        # renderer.render(geometry, next_states, input_force=input)
        renderer.render(geometry, simulation._system.kinematic, input_force=input, trajectory=trajectory)

        dt = clock.tick(60) / 1000


if __name__ == "__main__":
    main()
