import gym
import matplotlib.pyplot as plt
import numpy as np

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
    'epsilon': .4,
    'epsilon_decay': 1e-4,
}
n_episodes = 10_000  # total number of training episodes
n_episodes_log = 10  # number of episodes between logs
n_runs = 10  # number of runs used to evaluate the trained policy

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

# For each method
rl_methods = [
    Sarsa,
    QLearning,
]
for i, method_cls in enumerate(rl_methods, start=1):

    # Create method object
    match method_cls:
        case vb if vb is Sarsa or QLearning:
            starting_value = LinearApproxActionValue(n_features, env.action_space.n)
            kwargs = value_based_args
    method = method_cls(env, starting_value, **kwargs)
    print(method_cls.__name__)

    # Training
    episode = 0
    while episode < n_episodes:
        value, policy = method.train(n_episodes_log)
        episode += n_episodes_log
        print(episode)

        # Run some executions to evaluate policy
        steps = 0
        reached_goal = 0
        for j in range(n_runs):
            state = env.reset()
            while True:
                action = policy.sample_greedy(state)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                steps += 1
                if done:
                    if reward == 0:
                        reached_goal += 1
                    break

        # Print results
        print(f'   results over {n_runs} runs')
        print(f'      avg steps until done: {steps / n_runs:.2f}')
        print(f'      times reached goal: {reached_goal}')
        print()
