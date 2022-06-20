import gym
import matplotlib.pyplot as plt
import numpy as np

from policy import SoftmaxPolicy
from rl import *
from utils import FeaturizedEnv
from value import LinearApproxActionValue


# Variables of the example
max_episode_steps = 10_000  # max number of steps of the environment
n_features = 10_000  # number of features to be used
n_samples = 1_000  # samples used to fit the featurizer
features_type = 'rbf'  # type of features to be used
featurizer_args = {  # arguments of the featurizer
    'gamma': 1.,
}
value_based_args = {  # arguments of the value-based methods
    'gamma': .9,
    'alpha': .1,
    'alpha_decay': 1e-3,
    'epsilon': .8,
    'epsilon_decay': 1e-4,
    'epsilon_mode': 'exponential',
}
policy_based_args = {  # arguments of the policy-based methods
    'gamma': .9,
    'alpha': .1,
    'alpha_decay': 1e-3,
}
n_episodes = 10_000  # total number of episodes of training
n_runs = 50  # number of runs used to evaluate the trained policy

# Create environment
gym.envs.register(
    id='MyMountainCar',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=max_episode_steps,
)
base_env = gym.make('MyMountainCar')
env = FeaturizedEnv(
    env=base_env,
    n_features=n_features,
    n_samples=n_samples,
    features_type=features_type,
    featurizer_args=featurizer_args
)

vb_methods = [
    # Sarsa,
    # QLearning,
]
pb_methods = [
    # Reinforce,
    ActorCritic,
]

# For each method
for i, method_cls in enumerate(vb_methods + pb_methods, start=1):

    # Initialize method
    match method_cls:
        case vb if vb in vb_methods:
            starting_value = LinearApproxActionValue(n_features, env.action_space.n)
            kwargs = value_based_args
            method = method_cls(env, starting_value, verbose=True, **kwargs)
        case reinforce if reinforce is Reinforce:
            starting_policy = SoftmaxPolicy(n_features, env.action_space.n)
            kwargs = policy_based_args
            method = method_cls(env, starting_policy, verbose=True, **kwargs)
        case actor_critic if actor_critic is ActorCritic:
            starting_policy = SoftmaxPolicy(n_features, env.action_space.n)
            starting_value = LinearApproxActionValue(n_features, env.action_space.n)
            kwargs = policy_based_args
            method = method_cls(env, starting_policy, starting_value, verbose=True, **kwargs)

    print(method_cls.__name__)

    # Training
    episode = 0
    if method_cls in vb_methods:
        value, policy = method.train(n_episodes)
    else:
        policy = method.train(n_episodes)

    # Run some executions to evaluate policy
    reached_goal = 0
    for j in range(n_runs):
        steps = 0
        state = env.reset()
        while True:
            if method_cls in vb_methods: action = policy.sample_greedy(state)
            else: action = policy.sample(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            steps += 1
            if done:
                if steps != max_episode_steps:
                    reached_goal += 1
                break

    # Print results
    print(f'   results over {n_runs} runs')
    print(f'      times reached goal: {reached_goal}')
    print()
