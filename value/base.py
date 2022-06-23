__all__ = [
    'ActionValue',
]


import numpy.typing as npt


class ActionValue:
    """
    Base action-value function class.
    """

    def of(self, state, action: int) -> float:
        """
        Value of a given state or state-action pair.
        """
        raise NotImplementedError

    def all_values(self, state) -> npt.NDArray[float]:
        """
        All action values of a given state.
        """
        raise NotImplementedError

    def update(self, state, action: int, delta: float) -> None:
        """
        Update value of a given state-action pair.
        """
        raise NotImplementedError

    def to_array(self) -> npt.NDArray[float]:
        """
        Convert value to an array of state values.
        """
        raise NotImplementedError
