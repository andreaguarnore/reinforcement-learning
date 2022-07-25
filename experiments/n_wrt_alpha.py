from itertools import product

import gym
import numpy as np

from agents import *
from core.agent import Agent
from core.learning_rate import LearningRate
from core.value import TabularActionValue
from utils.experiment import MeanSquaredError


def epsilon_experiment() -> tuple[str, Agent]:
    alphas = {
        '0_1':  LearningRate('constant', 0.1),
    }
    ns = {
        '2': 2,
        '8': 8,
        '32': 32,
    }
    for (alpha_name, alpha), (n_name, n) in product(alphas.items(), ns.items()):
        name = f'alpha{alpha_name}_n{n_name}'
        yield name, nStepSarsa(
            env=env,
            starting_value=TabularActionValue(n_states, n_actions),
            gamma=gamma,
            epsilon=LearningRate('constant', 0.1),
            alpha=alpha,
            n=n,
        )


env = gym.make('CliffWalking-v0', new_step_api=True)
n_states = env.observation_space.n
n_actions = env.action_space.n

# Fix the environment's reward
rows = 4
cols = 12
raveled_state_to_goal = np.ravel_multi_index((2, 11), (rows, cols))
raveled_goal = np.ravel_multi_index((3, 11), (rows, cols))
env.P[raveled_state_to_goal][2] = [(1.0, raveled_goal, 0.0, True)]

gamma = 0.9
n_episodes = 10
experiment = MeanSquaredError(epsilon_experiment, env, gamma)
experiment.run(
    n_runs=100,
    episodes_to_log=list(range(1, n_episodes + 1, 1)),
)

import glob
import matplotlib.pyplot as plt
filenames = glob.glob('./*.dat')
for filename in filenames:
    errors = []
    with open(filename, 'r') as file:
        file.readline()  # skip header
        for line in file.readlines():
            _, mse, _, _ = line.split()
            errors.append(float(mse))
    plt.plot(np.arange(n_episodes) + 1, np.array(errors), label=filename)
plt.legend()
plt.xlabel('episodes')
plt.ylabel('mse')
plt.show()
