import gym
import numpy as np

from agents import *
from core.step_size import StepSize
from core.value import TabularActionValue
from utils.experiment import CumulativeReward, MeanSquaredError


agent_clss = QLearning
n_runs = 100
average_over = 10

env = gym.make('CliffWalking-v0', new_step_api=True)
n_states = env.observation_space.n
n_actions = env.action_space.n
gamma = 1.0

# Make the last state terminal
rows = 4
cols = 12
raveled_goal = np.ravel_multi_index((3, 11), (rows, cols))
for i in range(n_actions):
    env.P[raveled_goal][i] = [(1.0, raveled_goal, 0.0, True)]

# Create the agent and run the experiment on it
agent = agent_clss(
    env=env,
    initial_value=TabularActionValue(n_states, n_actions),
    gamma=gamma,
    epsilon=StepSize('constant', 0.1),
)
experiment = CumulativeReward(env)
rewards = experiment.run_experiment(
    agent=agent,
    n_runs=n_runs + average_over,
    episodes_to_log=range(1, 800),
    verbosity=1,
)

# Compute average over runs
rewards = np.mean(rewards, axis=1)

# Smooth via moving average
rewards = np.convolve(rewards, np.ones(average_over), 'valid') / average_over

# Save cumulative reward to file
with open(f'{agent_clss.__name__.lower()}_reward.dat', 'w') as file:
    file.write('episode reward\n')
    for episode, reward in enumerate(rewards):
        file.write(f'{episode} {np.mean(reward)}\n')

# Also compute MSE over some runs
print('computing mse')
experiment = MeanSquaredError(env, gamma)
episodes_to_log = range(1, 800, 3)
errors = experiment.run_experiment(
    agent=agent_clss(
        env=env,
        initial_value=TabularActionValue(n_states, n_actions),
        gamma=gamma,
        epsilon=StepSize('constant', 0.1),
    ),
    n_runs=n_runs,
    episodes_to_log=episodes_to_log,
    verbosity=1,
)

# Compute average over runs
errors = np.mean(errors, axis=1)

# Save MSE to file
with open(f'{agent_clss.__name__.lower()}_mse.dat', 'w') as file:
    file.write('episode mse\n')
    for episode, mse in zip(episodes_to_log, errors):
        file.write(f'{episode} {mse}\n')
