__all__ = [
    'first_visit_mc',
]


from collections import defaultdict
from copy import deepcopy

from gym import Env
import numpy as np

from common import TabularActionValue, DerivedPolicy


def first_visit_mc(
    env: Env,
    starting_value: TabularActionValue | None = None,
    gamma: float = .9,
    epsilon: float = .3,
    n_episodes: int = 10000,
    n_steps: int | None = None,
):
    """
    On-policy first-visit Monte Carlo control.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = TabularActionValue(n_states, n_actions)
    policy = DerivedPolicy(Q)

    returns = defaultdict(lambda: 0.)
    returns_count = defaultdict(lambda: 0)

    for _ in range(n_episodes):

        # Generate an episode following the policy
        episode = {'sa_pairs': [], 'rewards': []}
        state = env.reset()
        step = 0
        while n_steps is None or step < n_steps:

            action = policy.sample_epsilon_greedy(state, epsilon)
            next_state, reward, done, _ = env.step(action)

            episode['sa_pairs'].append((state, action))
            episode['rewards'].append(reward)

            if done:
                break
            state = next_state
            step += 1

        # For all visited state-action pairs in this episode
        sa_pairs = set(episode['sa_pairs'])
        for sa_pair in sa_pairs:

            # Find first occurrence in the episode
            fo_step = next(
                step for step, ep_sa_pair in enumerate(episode['sa_pairs']) \
                if sa_pair == ep_sa_pair
            )

            # Sum all rewards obtained since the first occurence
            G = sum([r * (gamma ** step) for step, r in enumerate(episode['rewards'][fo_step:])])

            # Compute average return and update counts
            returns[sa_pair] += G
            returns_count[sa_pair] += 1
            Q.update(*sa_pair, returns[sa_pair] / returns_count[sa_pair])

    return Q, policy
