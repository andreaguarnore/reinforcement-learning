__all__ = [
    'TabularStateValue',
    'ActionValue',
    'TabularActionValue',
    'LinearApproxStateValue',
    'LinearApproxActionValue',
]


import numpy as np

from core.step_size import StepSize


class TabularValue:
    """
    Generic tabular value function class.
    """

    def __init__(self, n_states: int, n_actions: int = None) -> None:
        self.n_states = n_states
        self.n_actions = n_actions

    def to_array(self) -> np.ndarray:
        """
        Return array representation of the value.
        """

class LinearApproxValue:
    """
    Generic class for a linearly approximate value function.
    """

    def __init__(
        self,
        n_features: np.ndarray,
        n_actions: int = None,
        lr: StepSize = None
    ) -> None:
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = StepSize(
            mode='constant',
            initial_step_size=1e-2,
        ) if lr is None else lr


class StateValue:
    """
    Generic state-value function class.
    """

    def of(self, state) -> float:
        """
        Value of a given state.
        """

    def update(self, state, update: float) -> None:
        """
        Update value of a given state.
        """


class ActionValue:
    """
    Generic action-value function class.
    """

    def of(self, state, action: int) -> float:
        """
        Value of a given state-action pair.
        """

    def all_values(self, state) -> np.ndarray:
        """
        All action values of a given state.
        """

    def update(self, state, action: int, update: float) -> None:
        """
        Update value of a given state-action pair
        """


class TabularStateValue(StateValue, TabularValue):
    """
    State-value function represented as a `(n_states,)` array.
    """

    def __init__(self, n_states: int) -> None:
        super().__init__(n_states)
        self.V = np.zeros(n_states)

    def of(self, state: int) -> float:
        return self.V[state]
    
    def update(self, state: int, update: float) -> None:
        self.V[state] += update

    def to_array(self) -> np.ndarray:
        return self.V.copy()


class TabularActionValue:
    """
    Action-value function represented as a `(n_states, n_actions)` array.
    """

    def __init__(self, n_states: int, n_actions: int) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.Q = np.zeros((n_states, n_actions))

    def of(self, state: int, action: int) -> float:
        return self.Q[state][action]

    def all_values(self, state: int) -> np.ndarray:
        return self.Q[state]

    def update(self, state: int, action: int, delta: float) -> None:
        self.Q[state][action] += delta

    def to_array(self) -> np.ndarray:
        return np.max(self.Q, axis=1)


class LinearApproxStateValue(StateValue, LinearApproxValue):
    """
    Linearly approximated state-value function.
    """

    def __init__(self, n_features: int, **kwargs) -> None:
        super().__init__(n_features, **kwargs)
        self.w = np.zeros(self.n_features)

    def of(self, features: np.ndarray) -> float:
        return self.w @ features

    def update(self, features: np.ndarray, update: float) -> None:
        self.w += self.lr() * update * features


class LinearApproxActionValue(ActionValue, LinearApproxValue):
    """
    Linearly approximated action-value function.
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int = None,
        **kwargs
    ) -> None:
        super().__init__(n_features, n_actions, **kwargs)
        self.w = np.zeros((self.n_features, self.n_actions))

    def of(self, features: np.ndarray, action: int) -> float:
        return self.w[:, action].T @ features

    def all_values(self, features: np.ndarray) -> np.ndarray:
        return self.w.T @ features

    def update(self, features: np.ndarray, action: int, update: float) -> None:
        self.w[:, action] += self.lr() * update * features
