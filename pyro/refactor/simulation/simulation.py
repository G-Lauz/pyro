import abc


class Simulation(abc.ABC):
    @abc.abstractmethod
    def run(self, dt: float = 0.01, steps: int = 1000, **kwargs):
        """
        Run the simulation for a given time and time step.

        :param dt: The time step of the simulation.                         (float)
        :param steps: The number of steps to run.                           (int)
        :param kwargs: Additional keyword arguments for specific simulation types.
        """
