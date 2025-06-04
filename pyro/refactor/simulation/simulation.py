import abc


class Simulation(abc.ABC):
    @abc.abstractmethod
    def run(self, dt: float = 0.01, steps: int = 1000):
        """
        Run the simulation for a given time and time step.

        :param time: The current time of the simulation.            (float)
        :param dt: The time step of the simulation.                 (float)
        """
