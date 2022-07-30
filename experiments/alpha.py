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

# For each alpha
experiment = MeanSquaredError(env, gamma)
alphas = {
    '0_01': StepSize('constant', 1e-2),
    '0_15': StepSize('constant', 0.15),
    '0_30': StepSize('constant', 0.30),
    '0_45': StepSize('constant', 0.45),
    '0_60': StepSize('constant', 0.60),
    '0_99': StepSize('constant', 0.99),
}
for name, alpha in alphas.items():

    print(name)

    # Run experiment
    episodes_to_log = range(1, 10_000, 50)
    error = experiment.run(
        agent = Sarsa(
            env=env,
            starting_value=TabularActionValue(n_states, n_actions),
            gamma=gamma,
            alpha=alpha,
            epsilon=StepSize('linear', 0.8, 1e-3),
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
