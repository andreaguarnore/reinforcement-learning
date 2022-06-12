__all__ = [
    'sarsa',
    'q_learning',
]


from copy import deepcopy

from gym import Env
import numpy as np

from common import ActionValue, LinearApproxActionValue, DerivedPolicy


def sarsa(
    env: Env,
    starting_value: ActionValue,
    gamma: float = .9,
    epsilon: float = .3,
    alpha: float = .1,
    decay: float = 1e-3,
    n_episodes: int = 10_000,
    n_steps: int | None = None,
):
    """
    On-policy temporal difference learning.
    """
    values = []
    Q = deepcopy(starting_value)
    policy = DerivedPolicy(Q)
    alpha0 = alpha

    # For each episode
    for episode in range(n_episodes):

        # Initialize S
        state = env.reset()

        # For each step of the episode
        step = 0
        while n_steps is None or step < n_steps:

            # Choose A from S epsilon greedily
            # using the policy derived from Q
            action = policy.sample_epsilon_greedy(state, epsilon)

            # Take action A, observe R, S'
            next_state, reward, done, _ = env.step(action)

            # Choose A' from S' epsilon greedily
            # using the policy derived from Q
            next_action = policy.sample_epsilon_greedy(next_state, epsilon)

            # TD update
            target = reward + gamma * Q.of(next_state, next_action)
            error = target - Q.of(state, action)
            update = alpha * error
            Q.update(state, action, update)

            if done:
                break
            state = next_state
            step += 1

        # Update learning rates
        alpha = alpha0 / (1 + decay * episode)
        if isinstance(Q, LinearApproxActionValue):
            Q.step()

    return Q, policy


def q_learning(
    env: Env,
    starting_value: ActionValue,
    gamma: float = .9,
    epsilon: float = .3,
    alpha: float = .1,
    decay: float = 1e-3,
    n_episodes: int = 10_000,
    n_steps: int | None = None,
):
    """
    Off-policy temporal difference learning.
    """
    Q = deepcopy(starting_value)
    policy = DerivedPolicy(Q)
    alpha0 = alpha

    # For each episode
    for episode in range(n_episodes):

        # Initialize S
        state = env.reset()

        # For each step of the episode
        step = 0
        while n_steps is None or step < n_steps:

            # Choose A from S epsilon greedily
            # using the policy derived from Q
            action = policy.sample_epsilon_greedy(state, epsilon)

            # Take action A, observe R, S'
            next_state, reward, done, _ = env.step(action)

            # Choose A' from S greedily
            # using the policy derived from Q
            next_action = policy.sample_greedy(next_state)

            # TD update
            target = reward + gamma * Q.of(next_state, next_action)
            error = target - Q.of(state, action)
            update = alpha * error
            Q.update(state, action, update)

            if done:
                break
            state = next_state
            step += 1

        # Update alpha
        alpha = alpha0 / (1 + decay * (episode + 1))
        if isinstance(Q, LinearApproxActionValue):
            Q.step()

    return Q, policy
