__all__ = [
    'ActionValue',
    'TabularActionValue',
    'DerivedPolicy',
]


import numpy as np


class ActionValue:
    """
    Base action-value function class.
    """

    def of(self, state: int, action: int) -> float:
        """
        Value of a given state or state-action pair.
        """
        raise NotImplementedError

    def all_values(self, state: int) -> np.ndarray:
        """
        All action values of a given state.
        """
        raise NotImplementedError

    def update(self, state: int, action: int, value: float) -> None:
        """
        Update value of a given state-action pair.
        """
        raise NotImplementedError

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

    def all_values(self, state: int) -> np.ndarray:
        return self.Q[state]

    def update(self, state: int, action: int, update: float) -> None:
        self.Q[state][action] += update

    def to_array(self) -> np.ndarray:
        """
        Convert value to a `(n_states,)` array of state values.
        """
        return np.max(self.Q, axis=1)


class DerivedPolicy:
    """
    Policy derived from an action-value function.
    """

    def __init__(self, Q: int) -> None:
        self.Q = Q
        self.n_actions = Q.n_actions

    def epsilon_probabilities(
        self,
        state: int,
        epsilon: float = .1,
    ) -> np.ndarray:
        """
        Return the epsilon probability distribution for the given state.
        """
        probs = np.ones(self.n_actions) * epsilon / self.n_actions
        best_action = self.sample_greedy(state)
        probs[best_action] += 1. - epsilon
        return probs

    def sample_epsilon_greedy(self, state: int, epsilon: float = .1) -> int:
        """
        Sample epsilon-greedily an action in the given state.
        """
        probs = self.epsilon_probabilities(state, epsilon)
        action = np.random.choice(probs.size, p=probs)
        return action

    def sample_greedy(self, state: int) -> int:
        """
        Greedily sample an action in the given state.
        """
        return np.argmax(self.Q.all_values(state))
