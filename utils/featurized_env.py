__all__ = [
    'FeaturizedEnv',
]


from gym import Env, ObservationWrapper
import numpy as np
import numpy.typing as npt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler


class FeaturizedEnv(ObservationWrapper):
    """
    Environment wrapper which transforms states into features.
    """

    def __init__(
        self,
        env: Env,
        n_features=100,
        n_samples=10_000,
        features_type: str = 'rbf',
        featurizer_args: dict = None,
    ) -> None:
        super().__init__(env)
        self.env = env
        if featurizer_args is None:
            featurizer_args = dict()
        match features_type:
            case 'rbf':
                features = RBFSampler(**featurizer_args, n_components=n_features)
        self.featurizer = Pipeline([
            ('scaler', StandardScaler()),
            (features_type, features),
        ])
        samples = np.array([self.env.observation_space.sample() for _ in range(n_samples)])
        self.featurizer.fit(samples)

    def observation(self, obs: float | tuple[float, ...]) -> npt.NDArray[float]:
        return self.featurizer.transform([obs]).squeeze()
