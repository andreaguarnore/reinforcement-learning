from itertools import product
from os.path import join
import sys

import gym
import numpy as np

from agents import *
from core.step_size import StepSize
from core.value import LinearApproxActionValue
from utils.experiment import Experiment, StepsPerEpisode
from utils.featurized_states import RadialBasisFunction


assert len(sys.argv) <= 2, 'Too many arguments'
assert len(sys.argv) == 1 or sys.argv[1] in ['Y', 'N'], 'Invalid argument'
to_train = len(sys.argv) == 1 or sys.argv[1] == 'Y'

gamma = 0.9
n_episodes = 500
n_runs = 100
n_eval_runs = 100
average_over = 10

eval_filename = 'eval_features.dat'
with open(eval_filename, 'w') as file:
    file.write('gamma n_features avg_steps least_steps most_steps\n')

base_env = gym.make('MountainCar-v0', max_episode_steps=10_000, new_step_api=True)
os = base_env.observation_space
n_actions = base_env.action_space.n

rbf_gammas = [
    5.0,
    10.0,
    25.0,
]
all_n_features = [
    100,
    500,
    1000,
]
for rbf_gamma, n_features in product(rbf_gammas, all_n_features):

    print(f'rbf gamma: {rbf_gamma}, # features: {n_features}')
    filename = str(int(rbf_gamma)) + '_gamma_' + str(n_features) + '_features'
    full_path = join('experiments', 'saved', filename + '.dat')

    if to_train:

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
        training_steps, eval_steps, eval_reward = experiment.run_experiment(
            agent=agent,
            episodes_to_log=range(1, n_episodes),
            n_runs=n_runs,
            verbosity=1,
        )
        experiment.dump_results(full_path)

    else:
        training_steps, eval_steps, _ = Experiment.load_results(full_path)

    # Compute moving average of training steps
    training_steps = np.mean(training_steps, axis=1)
    training_steps = np.convolve(
        training_steps,
        np.ones(average_over),
        'valid',
    ) / average_over

    # Save training
    with open(filename + '.dat', 'w') as file:
        file.write(f'episode steps\n')
        for episode, steps in enumerate(training_steps):
            file.write(f'{episode} {steps}\n')

    # Save evaluation
    with open(eval_filename, 'a') as file:
        file.write(f'{int(rbf_gamma)} {n_features} {np.mean(eval_steps)} {np.min(eval_steps)} {np.max(eval_steps)}\n')
