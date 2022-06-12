__all__ = [
    'LinearApproxActionValue',
    'FeaturizedEnv',
]


from gym import Env, ObservationWrapper
import numpy as np
from sklearn.base import BaseEstimator

from common import ActionValue


class LinearApproxActionValue(ActionValue):
    """
    Linearly approximated action-value function.
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        lr: float = .1,
        decay: float = 1e-2,
    ) -> None:
        self.n_actions = n_actions
        self.lr0 = lr
        self.lr = lr
        self.decay = decay
        self.episode = 0
        self.w = np.zeros((n_features, n_actions))

    def of(self, features: tuple[float, ...], action: int) -> float:
        return self.w[:, action].T @ features

    def all_values(self, features: tuple[float, ...]) -> np.ndarray:
        return self.w.T @ features

    def update(
        self,
        features: tuple[float, ...],
        action: int,
        update: float
    ) -> None:
        self.w[:, action] += self.lr * update * features

    def step(self) -> None:
        """
        Update learning rate when an episode has terminated.
        """
        self.lr = self.lr0 / (1 + self.decay * self.episode)
        self.episode += 1


class FeaturizedEnv(ObservationWrapper):
    """
    Environment wrapper which transforms states into features. Expects an
    `sklearn`-like fitted estimator with a `transform` method.
    """

    def __init__(self, env: Env, featurizer: BaseEstimator) -> None:
        super().__init__(env)
        self.env = env
        self.featurizer = featurizer

    def observation(self, obs: tuple[float, ...]) -> np.array:
        return self.featurizer.transform([obs]).squeeze()
