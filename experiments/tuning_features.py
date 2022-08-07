from itertools import product

import gym
import numpy as np

from agents import *
from core.step_size import StepSize
from core.value import LinearApproxActionValue
from utils.experiment import StepsPerEpisode
from utils.featurized_states import RadialBasisFunction


n_episodes = 500
gamma = 0.9
n_runs = 10
n_eval_runs = 100
filename = 'eval_features.dat'
with open(filename, 'w') as file:
    file.write('gamma n_features avg_steps least_steps most_steps\n')

gym.envs.register(
    id='ModifiedEnv',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=10_000,
)
base_env = gym.make('ModifiedEnv', new_step_api=True)
os = base_env.observation_space
n_actions = base_env.action_space.n

rbf_gammas = [5.0, 10.0, 25.0]
all_n_features = [100, 500, 1000]
for rbf_gamma, n_features in product(rbf_gammas, all_n_features):

    print(f'rbf gamma: {rbf_gamma}, # features: {n_features}')

    # Create environment
    env = RadialBasisFunction(
        env=base_env,
        limits=list(zip(os.low, os.high)),
        gamma=rbf_gamma,
        n_centers=n_features,
        new_step_api=True,
    )

    # Create agent
    agent = Sarsa(
        env=env,
        initial_value=LinearApproxActionValue(n_features, n_actions),
        gamma=gamma,
        alpha=StepSize('linear', 0.1, 1e-2),
        epsilon=StepSize('linear', 0.8, 1e-2),
    )

    # Run experiment
    experiment = StepsPerEpisode(env, n_eval_runs)
    eval_steps, eval_reward = experiment.run_experiment(
        agent=agent,
        episodes_to_log=range(1, n_episodes),
        n_runs=n_runs,
        verbosity=1,
    )

    # Save evaluation
    with open(filename, 'a') as file:
        file.write(f'{int(rbf_gamma)} {n_features} {np.mean(eval_steps)} {np.min(eval_steps)} {np.max(eval_steps)}\n')
