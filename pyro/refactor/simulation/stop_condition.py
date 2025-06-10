import abc

from pyro.refactor.system import System


class StopCondition(abc.ABC):
    """
    Abstract base class for stop conditions in simulations.
    """

    @abc.abstractmethod
    def is_met(self, system: System, **kwargs) -> bool:
        """
        Check if the stop condition is met.

        :param system: The system being simulated.
        :param kwargs: Additional keyword arguments for specific stop conditions.
        :return: True if the stop condition is met, False otherwise.
        """
