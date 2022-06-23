__all__ = [
    'ParameterizedPolicy',
]


import numpy.typing as npt

from utils import LearningRate, ConstantLR


class ParameterizedPolicy:
    """
    Policy modeled with a parameterized function.
    """

    def __init__(self, n_features: int, lr: LearningRate = None) -> None:
        self.n_features = n_features
        self.lr = ConstantLR(1e-2) if lr is None else lr

    def probabilities(self, features: npt.NDArray[float]) -> npt.NDArray[float]:
        """
        Return the probability distribution for the given features.
        """
        raise NotImplementedError

    def sample(self, features: npt.NDArray[float]) -> int | float:
        """
        Sample an action according to the given features.
        """
        raise NotImplementedError

    def update(
        self,
        features: npt.NDArray[float],
        action: int | float,
        update: float,
    ) -> None:
        """
        Update parameters of the policy.
        """
        raise NotImplementedError
