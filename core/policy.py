__all__ = [
    'TabularPolicy',
    'ImplicitPolicy',
    'ParameterizedPolicy',
    'SoftmaxPolicy',
    'GaussianPolicy',
]


import numpy as np

from core.learning_rate import LearningRate
from core.value import ActionValue


class TabularPolicy:
    """
    Stochastic policy represented as a `(n_states, n_actions)` array.
    """

    def __init__(self, n_states: int, n_actions: int) -> None:
        self.pi = np.ones((n_states, n_actions)) / n_actions

    def sample(self, state: int) -> int:
        """
        Sample an action according to the probability distribution in the given
        state.
        """
        probabilities = self.probabilities(state)
        action = np.random.choice(probabilities.size, p=probabilities)
        return action

    def sample_greedy(self, state: int) -> int:
        """
        Return the greedy action in the given state.
        """
        return np.argmax(self.pi[state])

    def probabilities(self, state: int) -> np.ndarray:
        """
        Return the probability distribution for the given state.
        """
        return self.pi[state]

    def make_deterministic(self, state: int, action: int) -> None:
        """
        Make policy deterministic in the given state.
        """
        self.pi[state] = np.eye(self.pi.shape[1])[action]


class ImplicitPolicy:
    """
    Epsilon-greedy policy implicitly derived from an action-value function.
    """

    def __init__(self, Q: ActionValue) -> None:
        self.Q = Q
        self.n_actions = Q.n_actions

    def epsilon_probabilities(self, state, epsilon: float) -> np.ndarray:
        """
        Return the epsilon probability distribution for the given state.
        """
        probabilities = np.ones(self.n_actions) * epsilon / self.n_actions
        best_action = self.sample_greedy(state)
        probabilities[best_action] += 1.0 - epsilon
        return probabilities

    def sample_epsilon_greedy(self, state, epsilon: float) -> int:
        """
        Sample an action epsilon-greedily.
        """
        probabilities = self.epsilon_probabilities(state, epsilon)
        action = np.random.choice(probabilities.size, p=probabilities)
        return action

    def sample_greedy(self, state) -> int:
        """
        Sample an action greedily.
        """
        return np.argmax(self.Q.all_values(state))


class ParameterizedPolicy:
    """
    Policy modeled with a parameterized function.
    """

    def __init__(self, n_features: int, lr: LearningRate = None) -> None:
        self.n_features = n_features
        self.lr = LearningRate('constant', 1e-2) if lr is None else lr

    def sample(self, features: np.ndarray) -> int | float:
        """
        Sample an action according to the probability distribution of the given
        features.
        """

    def update(
        self,
        features: np.ndarray,
        action: int | float,
        update: float
    ) -> None:
        """
        Update parameters of the policy.
        """


class SoftmaxPolicy(ParameterizedPolicy):

    def __init__(self, n_features: int, n_actions: int, **kwargs) -> None:
        super().__init__(n_features, **kwargs)
        self.n_actions = n_actions
        self.theta = np.zeros((self.n_features, n_actions))

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        e = np.exp(features.T @ self.theta)
        return e / np.sum(e)

    def sample(self, features: np.ndarray) -> int:
        probabilities = self.probabilities(features)
        action = np.random.choice(probabilities.size, p=probabilities)
        return action

    def update(self, features: np.ndarray, action: int, update: float) -> None:
        """
        Policy gradient: phi(s, a) - E[phi(s,·)]
        """
        phi_sa = np.zeros_like(self.theta)
        phi_sa[:, action] = features
        probabilities = self.probabilities(features)
        E_phi = np.tile(features, (self.n_actions, 1)).T * probabilities
        gradient = phi_sa - E_phi
        self.theta += self.lr.lr * update * gradient


class GaussianPolicy(ParameterizedPolicy):

    def __init__(self, n_features: int, **kwargs) -> None:
        super().__init__(n_features, **kwargs)
        self.theta_mu = np.zeros(n_features)
        self.theta_sigma = np.zeros(n_features)

    def mean(self, features: np.ndarray):
        return self.theta_mu.T @ features

    def std(self, features: np.ndarray):
        print(np.exp(self.theta_sigma.T @ features))
        return np.exp(self.theta_sigma.T @ features)

    def sample(self, features: np.ndarray) -> float:
        return np.random.normal(self.mean(features), self.std(features), (1,))

    def update(
        self,
        features: np.ndarray,
        action: float,
        update: float,
    ) -> None:
        mu_gradient = (action - self.mean(features) * features) / self.std(features)
        self.theta_mu += self.lr.lr * update * mu_gradient
        sigma_gradient = ((action - self.mean(features)) ** 2 / self.std(features) - 1) * features
        self.theta_sigma += self.lr.lr * update * sigma_gradient
