import gym
import matplotlib.pyplot as plt
import numpy as np

from methods import *


env = gym.make('FrozenLake-v1', is_slippery=True)
methods = [
    'policy_iteration',
    'value_iteration',
    # 'first_visit_mc'
]

# Metrics
n_runs = 1000
avg_steps = np.zeros(len(methods))
reached_goal = np.zeros(len(methods))

# Prepare plots
gridsize = (len(methods) // 4 + 1, min(len(methods), 3))
fig = plt.figure(figsize=(gridsize[0] * 9, gridsize[1] * 2.5))
rows = 4
cols = 4
x, y = np.meshgrid(np.arange(rows), np.arange(cols))

# For each method
for i, m in enumerate(methods, start=1):

    # Find optimal value
    V, policy = locals()[m](env)

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
    print(f'      avg steps: {steps / n_runs:.2f}')
    print(f'      times reached goal: {reached_goal}')
    print()

    # Convert value to surface and plot it
    z = V.to_array().reshape((rows, cols))
    ax = fig.add_subplot(*gridsize, i, projection='3d')
    ax.plot_surface(x, y, z)
    ax.set_xlabel('row')
    ax.set_ylabel('column')
    ax.set_zlabel('value')

plt.show()
