__all__ = [
    'TabularPolicy',
    'ImplicitPolicy',
    'ParameterizedPolicy',
    'SoftmaxPolicy',
    'GaussianPolicy',
]


import numpy as np

from core.step_size import StepSize
from core.value import ActionValue


class TabularPolicy:
    """
    Stochastic policy represented as a `(n_states, n_actions)` array.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        mode: str = 'uniform',
    ) -> None:
        assert mode in ['uniform', 'random'], 'Invalid policy initialization mode'
        match mode:
            case 'uniform':
                self.pi = np.ones((n_states, n_actions)) / n_actions
            case 'random':
                self.pi = np.random.rand(n_states, n_actions)
                self.pi /= np.sum(self.pi, axis=1)[:, None]

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
    Generic class for a policy modeled with a parameterized function.
    """

    def __init__(self, n_features: int, lr: StepSize = None) -> None:
        self.n_features = n_features
        self.lr = StepSize(
            mode='constant',
            initial_step_size=1e-2,
        ) if lr is None else lr

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
    """
    Softmax in action preferences.
    """

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
        Policy gradient: φ(s, a) - E[φ(s,·)]
        """
        phi_sa = np.zeros_like(self.theta)
        phi_sa[:, action] = features
        probabilities = self.probabilities(features)
        E_phi = np.tile(features, (self.n_actions, 1)).T * probabilities
        gradient = phi_sa - E_phi
        self.theta += self.lr() * update * gradient


class GaussianPolicy(ParameterizedPolicy):
    """
    Gaussian in action preferences. The standard deviation will be constant if
    passed in as argument.
    """

    def __init__(
        self,
        n_features: int,
        std_dev: float | None = None,
        **kwargs
    ) -> None:
        super().__init__(n_features, **kwargs)
        self.theta_mu = np.zeros(n_features)
        self.const_std_dev = std_dev is not None
        if self.const_std_dev:
            self.std_dev = std_dev
        else:
            self.theta_sigma = np.zeros(n_features)

    def mean(self, features: np.ndarray):
        return self.theta_mu.T @ features

    def std(self, features: np.ndarray):
        if self.const_std_dev: return self.std_dev
        return np.exp(self.theta_sigma.T @ features)

    def sample(self, features: np.ndarray) -> float:
        return np.random.normal(self.mean(features), self.std(features), (1,))

    def update(
        self,
        features: np.ndarray,
        action: float,
        update: float,
    ) -> None:
        """
        μ gradient: (a - μ(s)) / σ(s)²) φ(s)
        σ gradient: ((a - μ(s))² / σ(s)² - 1) φ(s)
        """
        mu_gradient = (action - self.mean(features)) * features / self.std(features) ** 2
        self.theta_mu += self.lr() * update * mu_gradient
        if not self.const_std_dev:
            sigma_gradient = ((action - self.mean(features)) ** 2 / (self.std(features)) ** 2 - 1) * features
            self.theta_sigma += self.lr() * update * sigma_gradient
