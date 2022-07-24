__all__ = [
    'Sarsa',
    'nStepSarsa',
    'SarsaLambda',
    'QLearning',
]


from itertools import product

from gym import Env
import numpy as np

from core.agent import ValueBasedAgent
from core.value import ActionValue, LinearApproxActionValue


class TDAgent(ValueBasedAgent):
    """
    TD(0) agent.
    """

    def td_update(
        self,
        state: int,
        action: int,
        error: float,
    ) -> None:
        self.Q.update(state, action, self.alpha() * error)

    def episode(self, n_steps: int | None = None) -> tuple[int, float]:

        # Sample the starting state
        state = self.env.reset()

        # For each step of the episode
        step = 0
        total_reward = 0.
        while n_steps is None or step < n_steps:

            # Sample an action epsilon-greedily
            action = self.pi.sample_epsilon_greedy(state, self.epsilon())

            # Sample the next state and
            # the reward associated with the last transition
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward

            # Sample the next action from the behavior policy
            next_action = self.next_action(next_state)

            # TD update
            target = reward + self.gamma * self.Q.of(next_state, next_action)
            error = target - self.Q.of(state, action)
            self.td_update(state, action, error)

            # Stop if the environment has terminated
            if terminated or truncated:
                break

            # Prepare for the next step
            state = next_state
            step += 1

        return step, total_reward


class Sarsa(TDAgent):
    """
    On-policy TD(0).
    """

    def next_action(self, next_state: int) -> int:
        return self.pi.sample_epsilon_greedy(next_state, self.epsilon())


class nStepSarsa(Sarsa):

    def __init__(
        self,
        env: Env,
        starting_value: ActionValue,
        n: int,
        **kwargs,
    ):
        assert n > 1, 'The n value for the n-step must be greater than 1'
        super().__init__(env, starting_value, **kwargs)
        self.n = n
        self.n_gammas = np.array([self.gamma ** step for step in range(self.n)])

    def episode(self, n_steps: int | None = None) -> tuple[int, float]:

        # Sample the starting state
        state = self.env.reset()

        # For each step of the episode
        step = 0
        total_reward = 0.
        last_n_rewards = np.zeros(0)
        while n_steps is None or step < n_steps:

            # Sample an action epsilon-greedily
            action = self.pi.sample_epsilon_greedy(state, self.epsilon())

            # Sample the next state and
            # the reward associated with the last transition
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            last_n_rewards = np.pad(
                array=last_n_rewards,
                pad_width=(0, 1),
                mode='constant',
                constant_values=reward,
            )

            # If we have sampled enough rewards
            if last_n_rewards.size == self.n:

                # Sample the next action from the behavior policy
                next_action = self.next_action(next_state)

                # Compute the n-step return
                Gn = np.sum(last_n_rewards * self.n_gammas)

                # Update value
                target = Gn + self.gamma ** self.n * self.Q.of(next_state, next_action)
                error = target - self.Q.of(state, action)
                self.td_update(state, action, error)

                # Remove oldest reward
                last_n_rewards = last_n_rewards[1:]

            # Stop if the environment has terminated
            if terminated or truncated:
                break

            # Prepare for the next step
            state = next_state
            step += 1

        return step, total_reward


class SarsaLambda(Sarsa):
    """
    Finite and approximate Sarsa(λ).
    """

    def __init__(
        self,
        env: Env,
        starting_value: ActionValue,
        lambda_: float = 0.9,
        **kwargs,
    ):
        super().__init__(env, starting_value, **kwargs)
        self.lambda_ = lambda_
        self.is_approximate = isinstance(self.Q, LinearApproxActionValue)
        self.td_update = self.approximate_td_update if self.is_approximate else self.finite_td_update

    def episode(self, n_steps: int | None = None) -> tuple[int, float]:
        self.traces = np.zeros((
            self.Q.n_features if self.is_approximate else self.Q.n_states,
            self.Q.n_actions,
        ))
        return super().episode(n_steps)

    def finite_td_update(
        self,
        state: int | np.ndarray,
        action: int,
        error: float,
    ) -> None:
        self.traces[state, action] += 1
        for s, a in product(range(self.Q.n_states), range(self.Q.n_actions)):
            self.Q.update(s, a, self.alpha() * error * self.traces[s, a])
            self.traces[s, a] *= self.gamma * self.lambda_

    def approximate_td_update(
        self,
        state: int | np.ndarray,
        action: int,
        error: float,
    ) -> None:
        self.traces = self.gamma * self.lambda_ * self.traces + state
        self.Q.update(state, action, self.alpha() * error * self.traces)


class QLearning(TDAgent):
    """
    Off-policy TD(0).
    """

    def next_action(self, next_state: int) -> int:
        return self.pi.sample_greedy(next_state)
