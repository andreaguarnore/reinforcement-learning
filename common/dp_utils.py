__all__ = [
    'one_step_lookahead',
    'TabularStateValue',
    'TabularPolicy',
]


from gym import Env
import numpy as np


def one_step_lookahead(
    env: Env,
    state: int,
    V: 'TabularStateValue',
    gamma: float = 0.9,
) -> np.ndarray:
    """
    Compute all action values in a given state.
    """
    n_actions = env.action_space.n
    action_values = np.zeros(n_actions)
    for action in range(n_actions):
        for trans_prob, next_state, reward, _ in env.P[state][action]:
            action_values[action] += trans_prob * (
                reward + gamma * V.of(next_state)
            )
    return action_values


class TabularStateValue:
    """
    Represent a state-value function as a `(n_states,)` array of values.
    """

    def __init__(self, n_states: int) -> None:
        self.n_states = n_states
        self.V = np.zeros(n_states)

    def of(self, state: int) -> float:
        """
        Get value of the given state.
        """
        return self.V[state]

    def update(self, state: int, new_value: float) -> None:
        """
        Update value of the given state.
        """
        self.V[state] = new_value

    def to_array(self) -> np.array:
        """
        Return the state-value function as a `(n_states,)` array.
        """
        return np.copy(self.V)


class TabularPolicy:
    """
    Represent a stochastic policy as a `(n_states, n_actions)` array.
    """

    def __init__(self, n_states: int, n_actions: int) -> None:
        self.policy = np.ones((n_states, n_actions)) / n_actions

    def probabilities(self, state: int) -> np.array:
        """
        Return the probability distribution for the given state.
        """
        return self.policy[state]

    def make_deterministic(self, state: int, action: int):
        """
        Make policy deterministic in the given state.
        """
        self.policy[state] = np.eye(self.policy.shape[1])[action]

    def sample(self, state: int) -> int:
        """
        Sample an action from a given state according to the probability
        distribution.
        """
        probabilities = self.probabilities(state)
        action = np.random.choice(len(probabilities), p=probabilities)
        return action

    def sample_greedy(self, state: int) -> int:
        """
        Return greedy action in the given state.
        """
        return np.argmax(self.policy[state])
