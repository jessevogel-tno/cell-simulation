from abc import abstractmethod

from risq.model import State


class Method:

    @abstractmethod
    def probability(self, time: int, state: State) -> float:
        """The probability that a cell is in given state at given time.
        In other words, the expected number of cells divided by"""
        pass

    @abstractmethod
    def variance(self, time: int, state: State) -> float:
        """The variance in the number of cells in given state at given time.
        Normalized by dividing by the number of cells.
        This makes sense becuase the variance is linear in random variables (if they are independent),
        so doubling the number of cells will roughly double the variance, so this quantity will stay constant.
        """
        pass

    @abstractmethod
    def name() -> str:
        """The name of the method, used in the legend of plots."""
        pass
