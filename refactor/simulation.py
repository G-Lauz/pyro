import abc

import numpy

from refactor.system import ContinuousDynamicSystem


class Simulation(abc.ABC):
    @abc.abstractmethod
    def step(self, input, dt=0.01):
        pass

    @abc.abstractmethod
    def run(self, controller, dt=0.01, steps=1000):
        pass


class ContinuousSimulation(Simulation):
    def __init__(self, system: ContinuousDynamicSystem):
        super().__init__()
        self._system = system
    
    def step(self, states, input, dt=0.01):
        return self._system.compute_dynamics(input, dt)

    def run(self, controller, dt=0.1, steps=1000):
        """
        Discrete time forward dynamics evaluation using the Euler integration method.

        \equation{begin}
            x_{k+1} = x_k + f(x_k, t_k) * dt
        \equation{end}

        :param state: The current state of the system.      (state_dimension, 1)
        :param input: The control input.                    (input_dimension, 1)
        :param time: The current time.                      (1, 1)
        :param dt: The time step.                           (1, 1)
        :param steps: The number of integration steps.      (1, 1)

        :return: The next state vector of the system.              (state_dimension, 1)
        """
        next_states = self._system.kinematic.states # TODO improve interface?

        for _ in range(steps):
            input = controller.compute_control(next_states, dt) # TODO
            next_states = self.step(input, dt)

        return next_states
