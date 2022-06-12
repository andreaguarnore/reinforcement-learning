__all__ = [
    'first_visit_mc',
    'off_policy_mc',
]


from copy import deepcopy

from gym import Env
import numpy as np

from common import TabularActionValue, DerivedPolicy


def first_visit_mc(
    env: Env,
    starting_value: TabularActionValue | None = None,
    gamma: float = .9,
    epsilon: float = .3,
    n_episodes: int = 10_000,
    n_steps: int | None = None,
):
    """
    On-policy first-visit Monte Carlo control.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = deepcopy(starting_value)
    policy = DerivedPolicy(Q)

    returns = np.zeros((n_states, n_actions))
    counter = np.zeros((n_states, n_actions), dtype=int)

    for _ in range(n_episodes):

        # Keep track of first occurrences of state-action pairs
        first_occurrences = dict()

        # Keep all rewards in the episode
        rewards = []

        # Generate an episode
        state = env.reset()
        step = 0
        while n_steps is None or step < n_steps:
            action = policy.sample_epsilon_greedy(state, epsilon)

            # Add state-action pair if it is its
            # first occurrence in this episode
            if (state, action) not in first_occurrences:
                first_occurrences[state, action] = step

            # Make new step
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
            state = next_state
            step += 1

        # For all visited state-action pairs
        for (state, action), fo_idx in first_occurrences.items():

            # Compute return starting from the first occurrence
            G = sum([r * gamma ** step for step, r in enumerate(rewards[fo_idx:])])

            # Update new value to the average return
            # of the state-action pair
            returns[state, action] += G
            counter[state, action] += 1
            update = (returns[state, action] / counter[state, action]) - Q.of(state, action)
            Q.update(state, action, update)

    return Q, policy


def off_policy_mc(
    env: Env,
    starting_value: TabularActionValue | None = None,
    gamma: float = .9,
    epsilon: float = .3,
    n_episodes: int = 10_000,
    n_steps: int | None = None,
):
    """
    Off-policy Monte Carlo control via importance sampling.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = deepcopy(starting_value)
    policy = DerivedPolicy(Q)

    # Cumulative sum of the weights given the first n returns
    C = np.zeros((n_states, n_actions))

    for _ in range(n_episodes):

        # Generate an episode following the behaviour policy
        episode = []
        state = env.reset()
        step = 0
        while n_steps is None or step < n_steps:
            action = policy.sample_epsilon_greedy(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
            step += 1

        # For each step of the episode,
        # starting from the last one
        G = 0
        W = 1
        for state, action, reward in reversed(episode):

            # Weight return according to similarity between policies
            G = gamma * G + reward
            C[state, action] += W

            # Update value towards corrected return
            update = (W / C[state, action]) * (G - Q.of(state, action))
            Q.update(state, action, update)

            # Stop looping if policies do not match
            greedy_action = policy.sample_greedy(state)
            if action != greedy_action:
                break

            # Update weights
            W = W / policy.epsilon_probabilities(state, epsilon)[action]

    return Q, policy