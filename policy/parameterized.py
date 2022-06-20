__all__ = [
    'ParameterizedPolicy',
]


import numpy.typing as npt


class ParameterizedPolicy:
    """
    Policy modeled with a parameterized function.
    """

    def probabilities(self, features: npt.NDArray[float]) -> npt.NDArray[float]:
        raise NotImplementedError

    def sample(self, features: npt.NDArray[float]) -> int | float:
        raise NotImplementedError

    def update(
        self,
        features: npt.NDArray[float],
        action: int | float,
        update: float
    ) -> None:
        raise NotImplementedError
