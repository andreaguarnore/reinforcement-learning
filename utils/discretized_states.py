__all__ = [
    'DiscretizedStates',
]


from gym import Env, ObservationWrapper
import numpy as np


class DiscretizedStates(ObservationWrapper):
    """
    Discretize states into bins.
    """

    def __init__(
        self,
        env: Env,
        n_bins: int | list[int],
        **kwargs
    ) -> None:
        super().__init__(env, **kwargs)
        os = self.env.observation_space
        assert len(os.shape) == 1, 'Unsupported observation space'
        dims = os.shape[0]
        if isinstance(n_bins, int):
            n_bins = [n_bins] * dims
        assert len(n_bins) == dims, 'The length of the list of number of bins does not match the dimensions of the observation space'
        self.n_bins = n_bins
        self.bins = []
        for dim, nb in enumerate(n_bins):
            self.bins.append(
                np.linspace(
                    os.low[dim],
                    os.high[dim],
                    num=nb,
                    endpoint=False
                )
            )

    def observation(self, obs: np.ndarray) -> int:
        idx = np.empty(obs.size, dtype=int)
        for dim, (b, o) in enumerate(zip(self.bins, obs)):
            idx[dim] = np.searchsorted(b, o) - 1
        return np.ravel_multi_index(idx, self.n_bins)
