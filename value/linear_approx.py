__all__ = [
    'LinearApproxActionValue',
]


import numpy as np
import numpy.typing as npt

from utils import LearningRate, ConstantLR
from value import ActionValue


class LinearApproxActionValue(ActionValue):
    """
    Linearly approximated action-value function.
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        lr: LearningRate = None,
    ) -> None:
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = ConstantLR(1e-2) if lr is None else lr
        self.w = np.zeros((n_features, n_actions))

    def of(self, features: npt.NDArray[float], action: int) -> float:
        return self.w[:, action].T @ features

    def all_values(self, features: npt.NDArray[float]) -> npt.NDArray[float]:
        return self.w.T @ features

    def update(
        self,
        features: npt.NDArray[float],
        action: int,
        delta: float
    ) -> None:
        self.w[:, action] += delta * features

    def to_array(self, meshgrid: npt.NDArray[float]) -> npt.NDArray[float]:
        return np.apply_along_axis(
            lambda _: np.max(self.all_values(_)), 2, meshgrid,
        )
