from gym import Env, ActionWrapper
import numpy as np


class DiscretizedActions(ActionWrapper):
    """
    Discretize continuous one-dimensional action space.
    """

    def __init__(self, env: Env, n_actions: int, **kwargs) -> None:
        super().__init__(env, **kwargs)
        assert n_actions > 1, 'The number of actions must be greater than 1'
        aspace = self.env.action_space
        assert len(aspace.shape) == 1, 'Unsupported action space'
        self.bins = np.linspace(aspace.low[0], aspace.high[0], num=n_actions, endpoint=False)

    def action(self, action: float):
        return np.searchsorted(self.bins, action, side='right') - 1
