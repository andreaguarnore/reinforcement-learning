import gym
from gym import Env, ObservationWrapper
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

from common import (
    FeaturizedEnv,
    TabularPolicy, TabularStateValue,
    TabularActionValue, LinearApproxActionValue,
)
from methods import *


# Variables of the example
max_episode_steps = 10_000  # max number of steps of the environment
n_features = 10_000  # number of features to be used
n_samples = 1_000  # samples used to fit the featurizer
quantiles = 30  # precision of the discretization of the environment
n_runs = 100  # number of runs used to evaluate the trained policy
n_tiles = 100  # precision of the plot for function approximation

# Environment wrapper which makes the states of
# the mountain car problem discrete with `quantiles` bins.
class MountainCarDiscrete(ObservationWrapper):

    def __init__(self, env: Env, quantiles: int = 300) -> None:
        super().__init__(env)
        self.env = env
        self.bins = []
        self.quantiles = quantiles
        os = self.env.observation_space
        for dim in range(os.shape[0]):
            self.bins.append(np.linspace(os.low[dim], os.high[dim], num=quantiles))

    def observation(self, obs: tuple[float, ...]) -> np.array:
        idx = 0
        for i, (b, o) in enumerate(zip(self.bins, obs)):
            idx += np.searchsorted(b, o) * (self.quantiles ** i)
        return idx

# Initialize default environment
gym.envs.register(
    id='MyMountainCar',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=max_episode_steps,
)
base_env = gym.make('MyMountainCar')

# Create and fit featurizer
# to the default environment
featurizer = Pipeline([
    ('scaler', StandardScaler()),
    ('rbf_sampler', RBFSampler(gamma=1., n_components=n_features)),
])
samples = np.array([base_env.observation_space.sample() for _ in range(n_samples)])
featurizer.fit(samples)

# Create wrapped environments
discrete_env = MountainCarDiscrete(base_env, quantiles)
sample_size = discrete_env.observation_space.shape[0]
continuous_states_env = FeaturizedEnv(base_env, featurizer)
envs = [
    # discrete_env,  # discrete states, discrete actions
    continuous_states_env,  # continuous states, discrete actions
]

# Choose methods to be evaluated
methods = [
    # 'policy_iteration',
    # 'value_iteration',
    'sarsa',
    'q_learning',
]

# For each environment type
from itertools import product
for env, method in product(envs, methods):

    # Create method's arguments
    match (env, method):
        case (e, 'policy_iteration') if e is discrete_env:
            args = (
                TabularPolicy(quantiles ** sample_size, env.action_space.n),
            )
            kwargs = {}
        case (e, 'value_iteration') if e is discrete_env:
            args = (
                TabularStateValue(quantiles ** sample_size),
            )
            kwargs = {}
        case (e, _) if e is discrete_env:
            args = (
                TabularActionValue(quantiles ** sample_size, env.action_space.n),
            )
            kwargs = {
                'epsilon': .3,
                'n_episodes': 1_000,
                'verbose': True,
            }
        case (e, 'sarsa' | 'q_learning') if e is continuous_states_env:
            args = (
                LinearApproxActionValue(n_features, env.action_space.n),
            )
            kwargs = {
                'epsilon': .3,
                'verbose': True,
            }
        case _:  # invalid env-method combination
            continue

    # Run method
    env_str = 'discrete' if env is discrete_env else 'continuous states'
    print(f'training for {env_str} environment with {method}')
    value, policy = locals()[method](env, *args, **kwargs)

    # Run some executions to evaluate policy
    print('evaluating learned policy')
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
                if steps != max_episode_steps:
                    reached_goal += 1
                break

    # Print results
    print(f'   results over {n_runs} runs')
    print(f'      avg steps until done: {steps / n_runs:.2f}')
    print(f'      times reached goal: {reached_goal}')
    print()

    # Compute surface
    if env is discrete_env:
        X, Y = np.meshgrid(np.arange(quantiles), np.arange(quantiles))
        cost_to_go = -value.to_array()
        Z = np.reshape(cost_to_go, (quantiles, quantiles))
    else:
        os = env.observation_space
        X, Y = np.meshgrid(
            np.linspace(os.low[0], os.high[0], num=n_tiles),
            np.linspace(os.low[1], os.high[1], num=n_tiles),
        )
        Z = np.apply_along_axis(
            lambda _: -np.max(
                # Q.wh[episode].T @ featurizer.transform([_]).squeeze()
                value.all_values(featurizer.transform([_]).squeeze())
            ),
            2,
            np.dstack([X, Y])
        )

    # Plot surface and save it
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    ax.set_zlabel('cost to go')
    ax.set_title(f'{method}, {env_str} environment')
    surface = ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)
    fig.savefig(f'examples/mountain_car_{env_str}_{method}.png')
