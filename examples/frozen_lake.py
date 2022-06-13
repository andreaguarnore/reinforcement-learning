import gym
import matplotlib.pyplot as plt
import numpy as np

from common import TabularPolicy, TabularStateValue, TabularActionValue
from methods import *


# Variables of the example
n_runs = 1_000  # number of runs used to evaluate the trained policy

env = gym.make('FrozenLake-v1', is_slippery=True)
methods = [
    'policy_iteration',
    'value_iteration',
    'first_visit_mc',
    'off_policy_mc',
    'sarsa',
    'q_learning',
]

# Prepare plots
gridsize = (1, len(methods)) if len(methods) <= 3 else \
    (2, 2) if len(methods) <= 4 else \
    (2, 3) if len(methods) <= 6 else (3, 3)
fig = plt.figure(figsize=(3 + gridsize[1] * 3, 2 + gridsize[0] * 1.75))
rows = 4
cols = 4
x, y = np.meshgrid(np.arange(rows), np.arange(cols))

# For each method
for i, m in enumerate(methods, start=1):

    # Initialize starting policy/value
    match m:
        case 'policy_iteration':
            n_states = env.observation_space.n
            n_actions = env.action_space.n
            args = (TabularPolicy(n_states, n_actions),)
            kwargs = {}
        case 'value_iteration':
            n_states = env.observation_space.n
            args = (TabularStateValue(n_states),)
            kwargs = {}
        case _:
            n_states = env.observation_space.n
            n_actions = env.action_space.n
            args = (TabularActionValue(n_states, n_actions),)
            kwargs = {
                'epsilon': .4,
                'n_episodes': 10_000,
            }

    # Find optimal value
    value, policy = locals()[m](env, *args, **kwargs)

    # Print resulting policy
    print(m)
    print('   policy:')
    for r in range(4):
        print('      ', end='')
        for c in range(4):
            action = policy.sample_greedy(r * cols + c)
            if action == 0: action_str = '← '
            elif action == 1: action_str = '↓ '
            elif action == 2: action_str = '→ '
            else: action_str = '↑ '
            print(action_str, end='')
        print()

    # Run some executions to evaluate policy
    steps = 0
    reached_goal = 0
    for j in range(n_runs):
        state = env.reset()
        while True:
            action = policy.sample_greedy(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            steps += 1
            if done:
                if reward == 1:
                    reached_goal += 1
                break

    # Print results
    print(f'   results over {n_runs} runs')
    print(f'      avg steps until done: {steps / n_runs:.2f}')
    print(f'      times reached goal: {reached_goal}')
    print()

    # Convert value to surface and plot it
    z = value.to_array().reshape((rows, cols))
    ax = fig.add_subplot(*gridsize, i, projection='3d')
    ax.plot_surface(x, y, z)
    ax.set_xlabel('row')
    ax.set_ylabel('column')
    ax.set_zlabel('value')
    ax.set_title(m)

plt.tight_layout()
plt.show()
