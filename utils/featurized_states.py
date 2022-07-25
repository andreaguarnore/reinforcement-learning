__all__ = [
    'RadialBasisFunction',
]


from gym import Env, ObservationWrapper
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler


class FeaturizedStates(ObservationWrapper):
    """
    Generic wrapper for featurized states as observation.
    """

    def __init__(
        self,
        env: Env,
        standardize: bool = True,
        n_samples: int = 10_000,
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        self.env = env
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('featurizer', self.featurizer),
        ]) if standardize else self.featurizer
        samples = np.array([self.env.observation_space.sample() for _ in range(n_samples)])
        self.pipeline.fit(samples)

    def observation(self, obs: float | np.ndarray) -> np.ndarray:
        return self.pipeline.transform([obs]).squeeze()


class RadialBasisFunction(FeaturizedStates):
    """
    Radial basis function as features of the state.
    """

    def __init__(
        self,
        env: Env,
        gamma: float = 1.0,
        n_features: int = 100,
        **kwargs,
    ) -> None:
        self.featurizer = RBFSampler(gamma=gamma, n_components=n_features)
        super().__init__(env, **kwargs)
