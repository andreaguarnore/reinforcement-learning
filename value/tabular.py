__all__ = [
    'TabularActionValue',
]


import numpy as np
import numpy.typing as npt

from value import ActionValue


class TabularActionValue(ActionValue):
    """
    Tabular action-value function class. Implemented as a matrix.
    """

    def __init__(self, n_states: int, n_actions: int) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.Q = np.zeros((n_states, n_actions))

    def of(self, state: int, action: int) -> float:
        return self.Q[state][action]

    def all_values(self, state: int) -> npt.NDArray[float]:
        return self.Q[state]

    def update(self, state: int, action: int, delta: float) -> None:
        self.Q[state][action] += delta

    def to_array(self) -> npt.NDArray[float]:
        return np.max(self.Q, axis=1)
