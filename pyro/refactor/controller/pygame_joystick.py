import numpy

from pyro.refactor.system import StaticSystem


class PygameJoystickController(StaticSystem):

    def __init__(self):
        super().__init__(name="Pygame Joystick Controller")

        self.add_output_port(name="u")

        # Lazy import pygame
        import pygame
        self.pygame = pygame

        if not self.pygame.get_init():
            self.pygame.init()

        self.pygame.joystick.init()
        self.joysticks = []

        self._initialize_joysticks()

    def _initialize_joysticks(self):
        self.joysticks = []
        joystick_count = self.pygame.joystick.get_count()
        for i in range(joystick_count):
            joystick = self.pygame.joystick.Joystick(i)
            joystick.init()
            self.joysticks.append(joystick)
            print(f"Initialized joystick: {joystick.get_name()}")

        if not self.joysticks:
            raise RuntimeError("No joysticks found. Please connect a joystick and try again.")


    def _input_handler(self):
        input_upper_bound = self.outputs["u"].upper_bounds  # TODO: better interface for inputs
        input_lower_bound = self.outputs["u"].lower_bounds

        input_force = numpy.zeros(2)

        for joystick in self.joysticks:
            # XBox 360 Left Stick (left -> right: axis 0, up -> down: axis 1)
            # See https://www.pygame.org/docs/ref/joystick.html#xbox-360-controller-pygame-2-x for more information
            # x_raw = joystick.get_axis(0)
            # y_raw = joystick.get_axis(1)
            # angle = numpy.arctan2(-x_raw, -y_raw)

            # thrust = (joystick.get_axis(5) + 1) / 2  # Right trigger for thrust [0, 1]

            # # Convert polar (thrust, angle) to Cartesian (x, y)
            # x_force = (thrust * numpy.cos(angle) * (input_upper_bound[0] - input_lower_bound[0]) * 0.5)
            # y_force = (thrust * numpy.sin(angle) * (input_upper_bound[1] - input_lower_bound[1]) * 0.5)
            # input_force = numpy.array([x_force, y_force])

            x_raw = joystick.get_axis(0)

            deadzone = 0.1
            if abs(x_raw) < deadzone:
                x_raw = 0.0

            thrust = (joystick.get_axis(5) + 1) / 2  # Right trigger for thrust [0, 1]

            target_surge = input_upper_bound[0] * thrust
            target_yaw_rate = x_raw * (input_upper_bound[1] - input_lower_bound[1]) * 0.5

            input_force = numpy.array([target_surge, target_yaw_rate])

        return input_force

    def compute_output(self, time: float, dt: float = 0.01) -> dict:
        input_force = self._input_handler()
        return {"u": input_force}
