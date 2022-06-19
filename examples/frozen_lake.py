from itertools import chain

import gym
import numpy as np

from examples.frozen_lake_utils import *
from mdp import (
    TabularPolicy, TabularStateValue,
    PolicyIteration, ValueIteration,
)
from rl import *
from value import TabularActionValue


# Variables of the example
dp_args = {  # arguments for dynamic programming methods
    'gamma': .9,
    'theta': 1e-3,
}
rl_args = {  # arguments for reinforcement learning methods
    'gamma': .9,
    'alpha': .1,
    'alpha_decay': 1e-3,
    'epsilon': .8,
    'epsilon_decay': 1e-4,
    'epsilon_mode': 'exponential',
}
train_episodes = 100  # training episodes between error computation
n_episodes = 10_000  # total number of episodes of training
n_runs = 1_000  # number of runs used to evaluate the trained policy

# Create environment
env = gym.make('FrozenLake-v1', is_slippery=True)
n_states = env.observation_space.n
n_actions = env.action_space.n
states_ignored = [5, 6, 11, 12, 15]  # states for which a policy is not useful

dp_methods = [
    PolicyIteration,
    ValueIteration,
]
mc_methods = [
    FirstVisitMC,
    EveryVisitMC,
    OffPolicyMC,
]
td_methods = [
    Sarsa,
    QLearning,
]
methods = dp_methods + mc_methods + td_methods
errors = []  # keep track of mse
values = []  # keep all final value functions

# For each method
for i, method_cls in enumerate(methods, start=1):

    # Initialize method
    match method_cls:
        case pi if pi is PolicyIteration:
            initial = TabularPolicy(n_states, n_actions)
            kwargs = dp_args
        case vi if vi is ValueIteration:
            n_states = env.observation_space.n
            initial = TabularStateValue(n_states)
            kwargs = dp_args
        case rl:
            starting_value = TabularActionValue(n_states, n_actions)
            kwargs = rl_args

    # Find optimal value
    if method_cls in dp_methods:
        method = method_cls(initial, n_states, n_actions, env.P, **kwargs)
        value, policy = method.solve()
        optimal_value = value.to_array()
    else:
        method = method_cls(env, starting_value, save_episodes=True, **kwargs)
        training_episodes = []
        method_errors = []  # this method's mse
        episode = 0
        while episode < n_episodes:
            value, policy, episodes = method.train(train_episodes)
            training_episodes += episodes
            mse = np.mean((value.to_array() - optimal_value) ** 2)
            method_errors.append(mse)
            episode += train_episodes
    values.append(value.to_array())

    # Evaluate training and learned policy
    print(method_cls.__name__)
    print_policy(policy, states_ignored)
    if method_cls not in dp_methods:
        eval_training(training_episodes)
        errors.append(method_errors)
    eval_episodes = eval_learned_policy(env, n_runs, policy)
    print()

plot_errors(mc_methods + td_methods, errors, train_episodes)
plot_values(methods, values)
