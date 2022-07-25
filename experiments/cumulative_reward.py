import gym
import numpy as np

from agents import *
from core.learning_rate import LearningRate
from core.value import TabularActionValue
from utils.experiment import CumulativeReward


env = gym.make('CliffWalking-v0', new_step_api=True)
n_states = env.observation_space.n
n_actions = env.action_space.n

# Fix the environment's reward
rows = 4
cols = 12
raveled_state_to_goal = np.ravel_multi_index((2, 11), (rows, cols))
raveled_goal = np.ravel_multi_index((3, 11), (rows, cols))
env.P[raveled_state_to_goal][2] = [(1.0, raveled_goal, 0.0, True)]

# Create the agent and run the experiment on it
gamma = 0.9
agent = Sarsa(
    env=env,
    starting_value=TabularActionValue(n_states, n_actions),
    gamma=gamma,
    epsilon=LearningRate('constant', 0.1),
)
experiment = CumulativeReward(agent, env)
agent = experiment.run(
    n_episodes=800,
    average_over=30,
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
