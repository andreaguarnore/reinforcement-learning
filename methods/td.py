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
    epsilon: float = .1,
    epsilon_decay: float = 1e-3,
    alpha: float = .1,
    alpha_decay: float = 1e-3,
    n_episodes: int = 200,
    n_steps: int | None = None,
    verbose: bool = False,
) -> tuple[ActionValue, DerivedPolicy]:
    """
    On-policy temporal difference learning.
    """
    Q = deepcopy(starting_value)
    policy = DerivedPolicy(Q)
    epsilon0 = epsilon
    alpha0 = alpha

    # For each episode
    for episode in range(n_episodes):

        # Initialize S
        state = env.reset()

        # For each step of the episode
        step = 0
        total_reward = 0.
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
            total_reward += reward

        # Update learning rates
        epsilon = epsilon0 / (1 + epsilon_decay * episode)
        alpha = alpha0 / (1 + alpha_decay * episode)
        if isinstance(Q, LinearApproxActionValue):
            Q.step()

        if verbose:
            print(f'episode {episode + 1}:')
            print(f'   steps: {step}')
            print(f'   total reward: {total_reward}')

    return Q, policy


def q_learning(
    env: Env,
    starting_value: ActionValue,
    gamma: float = .9,
    epsilon: float = .1,
    epsilon_decay: float = 1e-3,
    alpha: float = .1,
    alpha_decay: float = 1e-3,
    n_episodes: int = 200,
    n_steps: int | None = None,
    verbose: bool = False,
) -> tuple[ActionValue, DerivedPolicy]:
    """
    Off-policy temporal difference learning.
    """
    Q = deepcopy(starting_value)
    policy = DerivedPolicy(Q)
    epsilon0 = epsilon
    alpha0 = alpha

    # For each episode
    for episode in range(n_episodes):

        # Initialize S
        state = env.reset()

        # For each step of the episode
        step = 0
        total_reward = 0.
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
            total_reward += reward

        # Update learning rates
        epsilon = epsilon0 / (1 + epsilon_decay * episode)
        alpha = alpha0 / (1 + alpha_decay * episode)
        if isinstance(Q, LinearApproxActionValue):
            Q.step()

        if verbose:
            print(f'episode {episode + 1}:')
            print(f'   steps: {step}')
            print(f'   total reward: {total_reward}')

    return Q, policy
