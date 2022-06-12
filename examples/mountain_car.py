import gym
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

from common import FeaturizedEnv, LinearApproxActionValue
from methods import *


n_runs = 1_000

# Initialize default environment
gym.envs.register(
    id='MyMountainCar',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=10_000,
)
base_env = gym.make('MyMountainCar')

# Create and fit featurizer
# to the default environment
n_features = 100
featurizer = Pipeline([
    ('scaler', StandardScaler()),
    ('rbf_sampler', RBFSampler(gamma=1., n_components=n_features)),
])
n_samples = 1_000
samples = np.array([base_env.observation_space.sample() for _ in range(n_samples)])
featurizer.fit(samples)

# Create wrapped environment
env = FeaturizedEnv(base_env, featurizer)


methods = [
    # 'policy_iteration',
    # 'value_iteration',
    # 'first_visit_mc',
    # 'off_policy_mc',
    'sarsa',
    'q_learning',
]
starting_value = LinearApproxActionValue(n_features, env.action_space.n)

# For each method
for i, m in enumerate(methods, start=1):

    # Find optimal value
    value, policy = locals()[m](env, starting_value, n_episodes=1_000, epsilon=.1)

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
