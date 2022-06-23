__all__ = [
    'SoftmaxPolicy',
]


import numpy as np
import numpy.typing as npt

from policy import ParameterizedPolicy


class SoftmaxPolicy(ParameterizedPolicy):
    """
    Softmax in action preferences. Can only be used with discrete action spaces.
    """

    def __init__(self, n_features: int, n_actions: int, **kwargs) -> None:
        super().__init__(n_features, **kwargs)
        self.n_actions = n_actions
        self.theta = np.zeros(self.n_features * n_actions)

    def probabilities(self, features: npt.NDArray[float]) -> npt.NDArray[float]:
        theta_2d = np.reshape(self.theta, (self.n_features, self.n_actions), order='F')
        e = np.exp(features.T @ theta_2d)
        return e / np.sum(e)

    def sample(self, features: npt.NDArray[float]) -> int:
        probabilities = self.probabilities(features)
        return np.random.choice(probabilities.size, p=probabilities)

    def update(
        self,
        features: npt.NDArray[float],
        action: int,
        delta: float
    ) -> None:
        idxs = slice(self.n_features * action, self.n_features * (action + 1))
        phi_s = np.zeros_like(self.theta)
        phi_s[idxs] = features
        phi = np.tile(features, self.n_actions)
        probabilities = np.repeat(self.probabilities(features), self.n_features)
        self.theta += self.lr * delta * (phi_s - phi * probabilities)
