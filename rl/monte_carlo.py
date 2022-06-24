__all__ = [
    'FirstVisitMC',
    'EveryVisitMC',
    'OffPolicyMC',
]


from gym import Env
import numpy as np

from policy import DerivedPolicy
from rl import ValueBasedMethod
from utils import generate_episode
from value import ActionValue


class OnPolicyMC(ValueBasedMethod):

    def __init__(
        self,
        first_visit: bool,
        env: Env,
        starting_value: ActionValue,
        **kwargs,
    ) -> None:
        super().__init__(env, starting_value, **kwargs)
        self.first_visit = first_visit
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.returns = np.zeros((self.n_states, self.n_actions))
        self.counts = np.zeros((self.n_states, self.n_actions), dtype=int)

    def train_episode(self, n_steps: int | None = None) -> tuple[int, float]:

        # Generate an episode
        episode = generate_episode(
            self.env,
            self.pi.sample_epsilon_greedy,
            {'epsilon': self.epsilon.lr},
            n_steps
        )
        if self.save_episodes:
            self.file_logger.save_episode(episode)

        # Make a list of all visited state-action pairs for which we will compute an update
        if self.first_visit:
            visited = []
            first_visits = []
            for step, (state, action, reward) in enumerate(episode):
                if (state, action) not in first_visits:
                    first_visits.append((state, action))
                    visited.append((step, (state, action, reward)))
        else:
            visited = enumerate(episode)

        # For all visited state-action pairs
        for step, (state, action, reward) in visited:

            # Compute return starting from the step of the visit
            G = sum([r * self.gamma ** t for t, (_, _, r) in enumerate(episode[step:])])

            # Incrementally update value
            self.returns[state, action] += G
            self.counts[state, action] += 1
            exp_G = self.returns[state, action] / self.counts[state, action]
            update = self.alpha.lr * (exp_G - self.Q.of(state, action))  # use alpha != 1 only for non-stationary problems!
            self.Q.update(state, action, update)

        return len(episode), sum([r for _, _, r in episode])


class FirstVisitMC(OnPolicyMC):
    """
    On-policy first-visit Monte Carlo learning.
    """

    def __init__(
        self,
        env: Env,
        starting_value: ActionValue,
        **kwargs,
    ) -> None:
        super().__init__(
            first_visit=True,
            env=env,
            starting_value=starting_value,
            **kwargs,
        )


class EveryVisitMC(OnPolicyMC):
    """
    On-policy every-visit Monte Carlo learning.
    """

    def __init__(
        self,
        env: Env,
        starting_value: ActionValue,
        **kwargs,
    ) -> None:
        super().__init__(
            first_visit=False,
            env=env,
            starting_value=starting_value,
            **kwargs,
        )


class OffPolicyMC(ValueBasedMethod):
    """
    Off-policy Monte Carlo learning.
    """

    def __init__(
        self,
        env: Env,
        starting_value: ActionValue,
        **kwargs,
    ) -> None:
        super().__init__(env, starting_value, **kwargs)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n

        # Cumulative sum of the weights given the first `n` returns
        self.C = np.zeros((self.n_states, self.n_actions))

    def train_episode(self, n_steps: int | None = None) -> tuple[int, float]:

        # Generate an episode
        episode = generate_episode(
            self.env,
            self.pi.sample_epsilon_greedy,
            {'epsilon': self.epsilon.lr},
            n_steps,
        )
        if self.save_episodes:
            self.file_logger.save_episode(episode)

        # For each step of the episode, starting from the last one
        G = 0.
        W = 1.
        for state, action, reward in reversed(episode):

            # Weight return according to similarity between policies
            G = self.gamma * G + reward
            self.C[state, action] += W

            # Update value towards corrected return
            update = (W / self.C[state, action]) * self.alpha.lr * (G - self.Q.of(state, action))
            self.Q.update(state, action, update)

            # Stop looping if policies do not match
            greedy_action = self.pi.sample_greedy(state)
            if action != greedy_action:
                break

            # Update weights
            W = W / self.pi.epsilon_probabilities(state, self.epsilon.lr)[action]

        return len(episode), sum([r for _, _, r in episode])
