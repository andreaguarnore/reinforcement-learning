__all__ = [
    'Sarsa',
    'QLearning',
]


from itertools import product

import numpy as np
from gym import Env

from rl import ValueBasedMethod
from value import ActionValue, LinearApproxActionValue


class TDMethod(ValueBasedMethod):
    """
    Generic temporal difference learning method.
    """

    def __init__(
        self,
        env: Env,
        starting_value: ActionValue,
        greedy_next_action: bool,
        lambda_: float = .8,
        **kwargs,
    ):
        assert 0. <= lambda_ <= 1, 'The trace decay rate lambda must be in [0, 1]'
        super().__init__(env, starting_value, **kwargs)
        self.is_approx = isinstance(self.Q, LinearApproxActionValue)
        self.greedy_next_action = greedy_next_action
        self.lambda_ = lambda_

    def train_episode(self, n_steps: int | None = None) -> tuple[int, float]:

        # Initialize S
        state = self.env.reset()

        # Initialize eligibility traces
        if self.lambda_ != 0.:
            eligibility = np.zeros((
                self.Q.n_features if self.is_approx else self.Q.n_states,
                self.Q.n_actions,
            ))

        # For each step of the episode
        step = 0
        total_reward = 0.
        while n_steps is None or step < n_steps:

            # Choose A from S epsilon greedily using the policy derived from Q
            action = self.pi.sample_epsilon_greedy(state, self.epsilon.lr)

            # Take action A, observe R, S'
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            if self.save_episodes:
                self.file_logger.save_episode_step(state, action, reward)

            # Choose A' from S' using the behavior policy
            if self.greedy_next_action: next_action = self.pi.sample_greedy(next_state)
            else: next_action = self.pi.sample_epsilon_greedy(next_state, self.epsilon.lr)

            # TD update
            target = reward + self.gamma * self.Q.of(next_state, next_action)
            error = target - self.Q.of(state, action)
            match (self.lambda_, self.is_approx):
                case (0., _):  # TD(0)
                    self.Q.update(state, action, self.alpha.lr * error)
                case (_, True):  # TD(lambda) with value function approximation
                    eligibility = self.gamma * self.lambda_ * eligibility + state
                    self.Q.update(state, action, self.alpha.lr * error * eligibility)
                case (_, False):  # TD(lambda) with tabular value function
                    eligibility[state, action] += 1
                    for s, a in product(range(self.Q.n_states), range(self.Q.n_actions)):
                        self.Q.update(s, a, self.alpha.lr * error * eligibility[s, a])
                        eligibility[s, a] *= self.gamma * self.lambda_

            # Stop if the environment has terminated
            if done:
                break

            # Prepare for the next step
            state = next_state
            step += 1

        return step, total_reward


class Sarsa(TDMethod):
    """
    On-policy temporal difference learning.
    """

    def __init__(
        self,
        env: Env,
        starting_value: ActionValue,
        **kwargs,
    ) -> None:
        super().__init__(
            greedy_next_action=False,
            starting_value=starting_value,
            env=env,
            **kwargs,
        )


class QLearning(TDMethod):
    """
    Off-policy temporal difference learning.
    """

    def __init__(
        self,
        env: Env,
        starting_value: ActionValue,
        **kwargs,
    ) -> None:
        super().__init__(
            greedy_next_action=True,
            starting_value=starting_value,
            env=env,
            **kwargs,
        )
