import abc


class Simulation(abc.ABC):
    @abc.abstractmethod
    def step(self, input_force, dt=0.01):
        pass

    @abc.abstractmethod
    def run(self, controller, dt=0.01, steps=1000):
        pass
