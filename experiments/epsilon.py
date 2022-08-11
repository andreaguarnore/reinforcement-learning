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

n_runs = 100
gamma = 0.9

env = gym.make('FrozenLake-v1', is_slippery=True, new_step_api=True)
n_states = env.observation_space.n
n_actions = env.action_space.n

# For each agent
agent_classes = [
    FirstVisitMC,
    Sarsa,
    QLearning,
]
for agent_clss in agent_classes:

    agent_name = agent_clss.__name__.lower()

    # For each epsilon
    epsilons = {
        'constant_0_2': StepSize('constant', 0.2),
        'linear_1e-2':  StepSize('linear',   0.8, 1e-2),
        'linear_1e-3':  StepSize('linear',   0.8, 1e-3),
        'linear_1e-4':  StepSize('linear',   0.8, 1e-4),
    }
    for epsilon_name, epsilon in epsilons.items():

        filename = agent_name + '_' + epsilon_name
        full_path = join('experiments', 'saved', filename + '.dat')
        print(filename)
        episodes_to_log = range(1, 10_000, 50)

        # Run experiments if needed
        if to_train:
            experiment = MeanSquaredError(env, gamma)
            errors = experiment.run_experiment(
                agent=agent_clss(
                    env=env,
                    initial_value=TabularActionValue(n_states, n_actions),
                    gamma=gamma,
                    epsilon=epsilon,
                ),
                n_runs=n_runs,
                episodes_to_log=episodes_to_log,
                verbosity=1,
            )
            experiment.dump_results(full_path)
        else:
            errors = Experiment.load_results(full_path)

        # Save results to file
        with open(f'{filename}.dat', 'w') as file:
            file.write('episode mse top bottom\n')
            for episode, mse in zip(episodes_to_log, errors):
                mean = np.mean(mse)
                std = np.std(mse)
                file.write(f'{episode} {mean} {mean + std} {mean - std}\n')
