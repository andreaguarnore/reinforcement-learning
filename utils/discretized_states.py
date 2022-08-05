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
        assert len(n_bins) == dims, 'The length `n_bins` and the size of the observation space must match'
        self.n_bins = n_bins
        self.bins = []
        self.bin_sizes = []
        for dim, nb in enumerate(self.n_bins):
            assert nb > 0, 'The number of bins must be positive'
            self.bins.append(
                np.linspace(
                    os.low[dim],
                    os.high[dim],
                    num=nb,
                    endpoint=False,
                )
            )
            self.bin_sizes.append((os.high[dim] - os.low[dim]) / nb)

    def observation(self, obs: np.ndarray) -> int:
        idx = np.empty(obs.size, dtype=int)
        for dim, (b, o) in enumerate(zip(self.bins, obs)):
            idx[dim] = np.searchsorted(b, o, side='right') - 1
        return np.ravel_multi_index(idx, self.n_bins)

    def unravel_idx(self, idx: int) -> np.ndarray:
        """
        Return the midpoint of the bins in which the given index falls into.
        """
        return np.array([
            self.bins[dim][i] + self.bin_sizes[dim] / 2
            for dim, i in zip(
                range(len(self.n_bins)), np.unravel_index(idx, self.n_bins)
            )
        ])
