__all__ = [
    'ValueBasedMethod',
    'PolicyBasedMethod',
]


import os

from gym import Env
import numpy as np

from policy import DerivedPolicy, ParameterizedPolicy
from utils import Logger, LearningRate, LinearLR
from value import ActionValue, LinearApproxActionValue


class RLMethod:
    """
    Base reinforcement learning method class.

    Implementations of RL methods simply need to override `train_episode`, which is called in loop
    in the `train` method, and add method-specific learning rates to the list `lrs`.

    Beware that starting values/policies passed in are not copied.
    """

    def __init__(
        self,
        env: Env,
        verbose: bool = False,
        save_episodes: bool = False,
        gamma: float = .9,
        alpha: LinearLR = None,
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
        self.alpha = LinearLR() if alpha is None else alpha
        self.lrs = [self.alpha]
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
        assert n_episodes > 0, 'The number of episodes must be positive'
        assert n_steps is None or n_steps > 0, 'The number of steps must be positive'

        # For each episode
        for _ in range(n_episodes):

            # Train for an episode
            episode_steps, total_reward = self.train_episode(n_steps)

            # Log training data
            if self.verbose:
                self.logger.log_training_episode(self.total_episodes, episode_steps, total_reward)
            if self.save_episodes:
                self.file_logger.new_episode()

            # Prepare for the next episode
            self.total_episodes += 1
            for lr in self.lrs:
                lr.update(self.total_episodes)


class ValueBasedMethod(RLMethod):

    def __init__(
        self,
        env: Env,
        starting_value: ActionValue,
        epsilon: LearningRate = None,
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        self.Q = starting_value
        if isinstance(self.Q, LinearApproxActionValue):
            self.lrs.append(self.Q.lr)
        self.pi = DerivedPolicy(self.Q)
        self.epsilon = LinearLR() if epsilon is None else epsilon
        self.lrs.append(epsilon)

    def train(self, n_episodes: int, **kwargs) -> None:
        super().train(n_episodes, **kwargs)
        return self.Q, self.pi


class PolicyBasedMethod(RLMethod):

    def __init__(
        self,
        env: Env,
        starting_policy: ParameterizedPolicy,
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        self.pi = starting_policy
        self.lrs.append(self.pi.lr)

    def train(self, n_episodes: int, **kwargs) -> None:
        super().train(n_episodes, **kwargs)
        return self.pi
