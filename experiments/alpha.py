from os.path import join
import sys

import gym
import numpy as np

from agents import *
from core.step_size import StepSize
from core.value import TabularActionValue
from utils.experiment import Experiment, MeanSquaredError


assert len(sys.argv) <= 2, 'Too many arguments'
assert len(sys.argv) == 1 or sys.argv[1] in ['Y', 'N'], 'Invalid argument'
to_train = len(sys.argv) == 1 or sys.argv[1] == 'Y'

gamma = 0.9

env = gym.make('FrozenLake-v1', is_slippery=True, new_step_api=True)
n_states = env.observation_space.n
n_actions = env.action_space.n

# For each alpha
experiment = MeanSquaredError(env, gamma)
alphas = {
    '0_01': StepSize('constant', 1e-2),
    '0_15': StepSize('constant', 0.15),
    '0_30': StepSize('constant', 0.30),
}
for alpha_name, alpha in alphas.items():

    print(alpha_name)
    filename = alpha_name + '_alpha'
    full_path = join('experiments', 'saved', filename + '.dat')

    # Run experiment
    episodes_to_log = range(1, 10_000, 50)
    if to_train:
        errors = experiment.run_experiment(
            agent = Sarsa(
                env=env,
                initial_value=TabularActionValue(n_states, n_actions),
                gamma=gamma,
                alpha=alpha,
                epsilon=StepSize('linear', 0.8, 1e-3),
            ),
            n_runs=100,
            episodes_to_log=episodes_to_log,
            verbosity=1,
        )
        experiment.dump_results(full_path)
    else:
        errors = Experiment.load_results(full_path)

    # Save to file
    with open(f'{filename}.dat', 'w') as file:
        file.write('episode mse top bottom\n')
        for episode, mse in zip(episodes_to_log, errors):
            mean = np.mean(mse)
            std = np.std(mse)
            file.write(f'{episode} {mean} {mean + std} {mean - std}\n')
