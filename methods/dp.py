__all__ = [
    'policy_iteration',
    'value_iteration',
]


from copy import deepcopy

from gym import Env
import numpy as np

from common import (
    one_step_lookahead,
    TabularStateValue, TabularPolicy
)


def policy_iteration(
    env: Env,
    starting_policy: TabularPolicy | None = None,
    gamma: float = 0.9,
    theta: float = 1e-4,
) -> tuple[TabularStateValue, TabularPolicy]:
    """
    Model-based control algorithm.

    Prediction: Bellman expectation backup.
    Control: Greedy policy improvement.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy = deepcopy(starting_policy) if starting_policy is not None \
        else TabularPolicy(n_states, n_actions)

    while True:

        # Evaluate current policy
        V = iterative_policy_evaluation(env, policy, gamma, theta)

        # Improve policy
        policy, policy_stable = policy_improvement(env, policy, V, gamma)

        # If no changes to the policy have been made,
        # the current policy is optimal
        if policy_stable:
            return V, policy


def iterative_policy_evaluation(
    env: Env,
    policy: TabularPolicy,
    gamma: float = 0.9,
    theta: float = 1e-4,
) -> TabularStateValue:
    """
    Return value approximated to true value of the given policy.
    """
    n_states = env.observation_space.n
    V = TabularStateValue(n_states)

    # Until the approximation is accurate enough
    while True:

        delta = 0.
        for state in range(n_states):

            # Compute the value of this state
            state_value = 0.
            for action, action_prob in enumerate(policy.probabilities(state)):
                for trans_prob, next_state, reward, _ in env.P[state][action]:
                    state_value += action_prob * trans_prob * (
                        reward + gamma * V.of(next_state)
                    )

            # Update the value function
            delta = max(delta, abs(state_value - V.of(state)))
            V.update(state, state_value)

        if delta < theta:
            return V


def policy_improvement(
    env: Env,
    policy: TabularPolicy,
    V: TabularStateValue,
    gamma: float = 0.9
) -> tuple[TabularPolicy, bool]:
    """
    Improve policy towards optimality.
    """
    n_states = env.observation_space.n
    policy_stable = True
    for state in range(n_states):

        # Find the best action under the current policy
        old_action = policy.sample_greedy(state)

        # Find the best action according to the current value
        best_action = np.argmax(one_step_lookahead(env, state, V, gamma))

        # Update the policy
        if old_action != best_action:
            policy_stable = False
        policy.make_deterministic(state, best_action)

    return policy, policy_stable


def value_iteration(
    env: Env,
    starting_value: TabularStateValue | None = None,
    gamma: float = 0.9,
    theta: float = 1e-4,
) -> tuple[TabularStateValue, TabularPolicy]:
    """
    Model-based control algorithm.

    Prediction and control: Bellman optimality backup.
    """
    n_states = env.observation_space.n
    V = deepcopy(starting_value) if starting_value is not None \
        else TabularStateValue(n_states)

    # Until the approximation is accurate enough
    while True:

        delta = 0.
        for state in range(n_states):

            # Find maximum action-value
            action_values = one_step_lookahead(env, state, V, gamma)
            max_action_value = np.max(action_values)

            # Update the value function
            delta = max(delta, abs(max_action_value - V.of(state)))
            V.update(state, max_action_value)

        if delta < theta:
            break

    # Compute corresponding deterministic policy
    n_actions = env.action_space.n
    policy = TabularPolicy(n_states, n_actions)
    for state in range(n_states):
        action_values = one_step_lookahead(env, state, V, gamma)
        best_action = np.argmax(action_values)
        policy.make_deterministic(state, best_action)

    return V, policy
