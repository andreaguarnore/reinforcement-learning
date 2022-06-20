__all__ = [
    'ValueBasedMethod',
    'PolicyBasedMethod',
]


from copy import deepcopy
import os

from gym import Env
import numpy as np

from policy import DerivedPolicy, ParameterizedPolicy
from utils import Logger
from value import ActionValue


class Method:
    """
    Base reinforcement learning method class. Implementations of RL methods simply need to override
    `train_episode`, which is called in loop in the `train` method.
    """

    def __init__(
        self,
        env: Env,
        verbose: bool = False,
        save_episodes: bool = False,
        gamma: float = .9,
        alpha: float = .1,
        alpha_decay: float = 1e-3,
    ) -> None:
        self.env = env
        self.verbose = verbose
        self.save_episodes = save_episodes
        if verbose:
            self.logger = Logger(f'{hash(self)}_info')
        if self.save_episodes:
            if not os.path.exists('logs'):
                os.makedirs('logs')
            self.file_logger = Logger(f'{hash(self)}_episodes', f'logs/{type(self).__name__}.log')
        self.gamma = gamma
        self.lrs = {
            'alpha': (alpha, alpha_decay, 'robbins_monro'),
        }
        self.total_episodes = 0

    def train_episode(self, n_steps: int | None = None) -> tuple[int, float]:
        """
        Train for one episode. Should never be called directly, but only through the `train` method.
        Returns the length of the episode and the total reward obtained during the episode.
        """
        raise NotImplementedError

    def train(self, n_episodes: int, n_steps: int | None = None) -> None:
        """
        Train for `n_episodes` episodes.
        """
        for _ in range(n_episodes):
            self.update_lrs(self.total_episodes)
            episode_steps, total_reward = self.train_episode(n_steps)
            self.total_episodes += 1
            if self.verbose:
                self.logger.log_training_episode(self.total_episodes, episode_steps, total_reward)
            if self.save_episodes:
                self.file_logger.new_episode()

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
        super().train(n_episodes, **kwargs)
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
        super().train(n_episodes, **kwargs)
        return self.pi
