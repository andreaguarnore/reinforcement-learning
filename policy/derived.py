__all__ = [
    'DerivedPolicy',
]


import numpy as np
import numpy.typing as npt

from value import ActionValue


class DerivedPolicy:
    """
    Policy derived from an action-value function.
    """

    def __init__(self, Q: 'ActionValue') -> None:
        self.Q = Q
        self.n_actions = Q.n_actions

    def epsilon_probabilities(self, state, epsilon: float = .1) -> npt.NDArray[float]:
        """
        Return the epsilon probability distribution for the given state.
        """
        probs = np.ones(self.n_actions) * epsilon / self.n_actions
        best_action = self.sample_greedy(state)
        probs[best_action] += 1. - epsilon
        return probs

    def sample_epsilon_greedy(self, state, epsilon: float = .1) -> int:
        """
        Sample epsilon-greedily an action in the given state.
        """
        probs = self.epsilon_probabilities(state, epsilon)
        action = np.random.choice(probs.size, p=probs)
        return action

    def sample_greedy(self, state) -> int:
        """
        Greedily sample an action in the given state.
        """
        return np.argmax(self.Q.all_values(state))
