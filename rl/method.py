__all__ = [
    'ValueBasedMethod',
    'PolicyBasedMethod',
]


from copy import deepcopy

from gym import Env
import numpy as np

from policy import DerivedPolicy, ParameterizedPolicy
from value import ActionValue


class Method:
    """
    Base reinforcement learning method class. Implementations of RL methods simply need to override
    `train_episode`, which is called in loop in the `train` method.
    """

    def __init__(
        self,
        env: Env,
        save_episodes = False,
        gamma: float = .9,
        alpha: float = .1,
        alpha_decay: float = 1e-3,
    ) -> None:
        self.env = env
        self.save_episodes = save_episodes
        self.gamma = gamma
        self.lrs = {
            'alpha': (alpha, alpha_decay, 'robbins_monro'),
        }
        self.total_episodes = 0

    def train_episode(
        self,
        n_steps: int | None = None,
        save_episode: bool = False,
    ) -> None | list[tuple[int | float, int | float, float]]:
        """
        Train for one episode. Should never be called directly, but only through the `train` method.
        Returns a list of `(state, action, reward)` tuples if `save_episode` is true.
        """
        raise NotImplementedError

    def train(
        self,
        n_episodes: int,
        n_steps: int | None = None,
    ) -> None | list[list[tuple[int | float, int | float, float]], ...]:
        """
        Train for `n_episodes` episodes.
        """
        if self.save_episodes:
            saved_episodes = []
        for _ in range(n_episodes):
            self.update_lrs(self.total_episodes)
            episode = self.train_episode(n_steps, self.save_episodes)
            if self.save_episodes:
                saved_episodes.append(episode)
            self.total_episodes += 1
        if self.save_episodes:
            return saved_episodes

    def update_lrs(self, episode: int) -> None:
        """
        Update learning rates each episode.
        """
        for lr, (lr0, decay, mode) in self.lrs.items():
            match mode:
                case 'constant':
                    self.__dict__[lr] = lr0
                case 'exponential':
                    self.__dict__[lr] = lr0 * np.exp(-decay * episode)
                case 'robbins_monro':
                    self.__dict__[lr] = lr0 / (1. + (decay * episode))


class ValueBasedMethod(Method):

    def __init__(
        self,
        env: Env,
        starting_value: ActionValue,
        epsilon: float = .3,
        epsilon_decay: float = .1,
        epsilon_mode: str = 'exponential',
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        self.Q = deepcopy(starting_value)
        self.pi = DerivedPolicy(self.Q)
        self.lrs.update({
            'epsilon': (epsilon, epsilon_decay, epsilon_mode),
        })

    def train(self, n_episodes: int, **kwargs) -> None:
        saved_episodes = super().train(n_episodes, **kwargs)
        if self.save_episodes:
            return self.Q, self.pi, saved_episodes
        return self.Q, self.pi


class PolicyBasedMethod(Method):

    def __init__(
        self,
        env: Env,
        starting_policy: ParameterizedPolicy,
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        self.pi = deepcopy(starting_policy)

    def train(self, n_episodes: int, **kwargs) -> None:
        saved_episodes = super().train(n_episodes, **kwargs)
        if self.save_episodes:
            return self.pi, saved_episodes
        return self.pi
