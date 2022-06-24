from copy import deepcopy

import gym
import matplotlib.pyplot as plt
import numpy as np

from mdp import TabularPolicy, TabularStateValue, ValueIteration
from rl import *
from utils.learning_rate import *
from value import TabularActionValue


# Variables of the example
gamma = .9  # keep it the same for all methods
mc_args = {  # arguments for monte carlo methods
    'gamma': gamma,
    'alpha': ConstantLR(1.),
    'epsilon': LinearLR(lr0=.8, decay=1e-5),
}
td_args = {  # arguments for temporal difference methods
    'gamma': gamma,
    'alpha': LinearLR(),
    'epsilon': LinearLR(lr0=.8, decay=1e-5),
    'lambda_': .1,
}
train_episodes = 1_000  # training episodes between error computation
threshold = 1e-3  # precision of the estimate of the optimal value function

# Create environment
env = gym.make('FrozenLake-v1', is_slippery=True)
n_states = env.observation_space.n
n_actions = env.action_space.n

mc_methods = [
    FirstVisitMC,
    EveryVisitMC,
    OffPolicyMC,
]
td_methods = [
    Sarsa,
    QLearning,
]
errors = []  # keep track of mean squared errors for each method

# Find optimal value by using a dynamic programming method
optimal_value, _ = ValueIteration(
    TabularStateValue(n_states),
    n_states,
    n_actions,
    env.P,
    gamma=gamma,
    theta=1e-3,
).solve()
optimal_value = optimal_value.to_array()

# For each method
methods = mc_methods + td_methods
for i, method_cls in enumerate(methods, start=1):

    # Initialize method
    method_name = method_cls.__name__
    print(method_name)
    starting_value = TabularActionValue(n_states, n_actions)
    kwargs = deepcopy(mc_args if method_cls in mc_methods else td_args)
    method = method_cls(env, starting_value, **kwargs)

    # Every `train_episodes` compute the mean squared error between the optimal value computed with
    # a dynamic programming method and the current value of the reinforcement learning method
    file = open(f'experiments/mse_{method_name}.log', 'w')
    method_errors = []  # this method's mse
    mse = float('inf')
    episode = 0
    while mse > threshold:
        value, policy = method.train(train_episodes)
        mse = np.mean((value.to_array() - optimal_value) ** 2)
        file.write(f'{episode} {mse}\n')
        method_errors.append(mse)
        episode += train_episodes
    errors.append(method_errors)
    file.close()

# Plot errors and compute value functions
fig = plt.figure()
for method, mse in zip(methods, errors):
    plt.plot(np.arange(len(mse)) * train_episodes, mse, label=method)

plt.legend()
plt.xscale('log')
# plt.yscale('log')
plt.tight_layout()
plt.show()
