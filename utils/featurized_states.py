from gym import Env, ObservationWrapper
import numpy as np


class FeaturizedStates(ObservationWrapper):
    """
    Generic wrapper for featurized states as observation.
    """

    def __init__(self, env: Env, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.env = env

    def observation(self, obs: float | np.ndarray) -> np.ndarray:
        return self.transform(obs)


class RadialBasisFunction(FeaturizedStates):
    """
    Radial basis function as features of the state.
    """

    def __init__(
        self,
        env: Env,
        limits: list[tuple[float, float]],
        gamma: float = 1.0,
        n_centers: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        assert gamma > 0.0, 'The RBF parameter gamma must be greater than zero'
        self.gamma = gamma
        self.centers = np.random.rand(n_centers, len(limits))
        for d, (min, max) in enumerate(limits):
            assert min < max, 'Invalid limits'
            width = max - min
            self.centers[:, d] = self.centers[:, d] * width + min

    def transform(self, obs: float | np.ndarray) -> np.ndarray:
        return np.exp(-(self.gamma * np.linalg.norm(obs - self.centers, axis=1)) ** 2.0)
