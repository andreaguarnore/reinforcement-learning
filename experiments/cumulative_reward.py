import gym
import numpy as np

from agents import *
from core.step_size import StepSize
from core.value import TabularActionValue
from utils.experiment import CumulativeReward, MeanSquaredError


agent_clss = Sarsa
verbose = True

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
    starting_value=TabularActionValue(n_states, n_actions),
    gamma=gamma,
    epsilon=StepSize('constant', 0.1),
)
experiment = CumulativeReward(agent, env)
agent = experiment.run(
    n_episodes=800,
    average_over=30,
    verbose=verbose,
)

# Print the learned policy
policy = agent.policy
for row in range(rows):
    for col in range(cols):
        state = np.ravel_multi_index((row, col), (rows, cols))
        match policy.sample_greedy(state):
            case 0: print('↑', end=' ')
            case 1: print('→', end=' ')
            case 2: print('↓', end=' ')
            case 3: print('←', end=' ')
    print()

# Also compute MSE over some runs
print('computing mse')
experiment = MeanSquaredError(env, gamma)
episodes_to_log = range(1, 800, 3)
error = experiment.run(
    agent = agent_clss(
        env=env,
        starting_value=TabularActionValue(n_states, n_actions),
        gamma=gamma,
        # epsilon=LearningRate('constant', 0.1),
        epsilon=StepSize('linear', 1e-2),
    ),
    n_runs=10,
    episodes_to_log=list(episodes_to_log),
    verbose=verbose,
)

# Save to file
with open(f'mse.dat', 'w') as file:
    file.write('episode mse\n')
    for episode, mse in zip(episodes_to_log, error):
        file.write(f'{episode} {np.mean(mse)}\n')
