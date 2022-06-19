__all__ = [
    'LinearApproxActionValue',
]


import numpy as np
import numpy.typing as npt

from value import ActionValue


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

    def of(self, features: npt.NDArray[float], action: int) -> float:
        return self.w[:, action].T @ features

    def all_values(self, features: npt.NDArray[float]) -> npt.NDArray[float]:
        return self.w.T @ features

    def update(
        self,
        features: npt.NDArray[float],
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

    def to_array(self, meshgrid: npt.NDArray[float]) -> npt.NDArray[float]:
        return np.apply_along_axis(
            lambda _: np.max(self.all_values(_)), 2, meshgrid,
        )
