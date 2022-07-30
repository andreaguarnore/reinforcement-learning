import gym
import numpy as np

from agents import *
from core.step_size import StepSize
from core.value import TabularActionValue
from utils.experiment import MeanSquaredError


env = gym.make('FrozenLake-v1', is_slippery=True, new_step_api=True)
n_states = env.observation_space.n
n_actions = env.action_space.n
gamma = 0.9

# For each epsilon
experiment = MeanSquaredError(env, gamma)
epsilons = {
    'constant_0_2': StepSize('constant', 0.2),
    'linear_1e-2':  StepSize('linear',   0.8, 1e-2),
    'linear_1e-3':  StepSize('linear',   0.8, 1e-3),
    'linear_1e-4':  StepSize('linear',   0.8, 1e-4),
}
for name, epsilon in epsilons.items():

    print(name)

    # Run experiment
    episodes_to_log = range(1, 10_000, 50)
    error = experiment.run(
        agent = QLearning(
            env=env,
            starting_value=TabularActionValue(n_states, n_actions),
            gamma=gamma,
            epsilon=epsilon,
        ),
        n_runs=10,
        episodes_to_log=list(episodes_to_log),
    )

    # Save to file
    with open(f'{name}.dat', 'w') as file:
        file.write('episode mse top bottom\n')
        for episode, mse in zip(episodes_to_log, error):
            mean = np.mean(mse)
            std = np.std(mse)
            file.write(f'{episode} {mean} {mean + std} {mean - std}\n')
