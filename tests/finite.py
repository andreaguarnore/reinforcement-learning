import gym

from agents import *
from core.value import TabularActionValue


# Print the policy for each state of the lake.
def print_policy(policy):
    holes = [5, 7, 11, 12, 15]
    rows = 4
    cols = 4
    for r in range(rows):
        for c in range(cols):
            state = r * cols + c
            if state not in holes:
                action = policy.sample_greedy(state)
                if action == 0: action_str = '← '
                elif action == 1: action_str = '↓ '
                elif action == 2: action_str = '→ '
                else: action_str = '↑ '
                print(action_str, end='')
            else:
                print('- ', end='')
        print()
    print()

n_episodes = 10_000

# Create environment
env = gym.make('FrozenLake-v1', is_slippery=False, new_step_api=True)
n_states = env.observation_space.n
n_actions = env.action_space.n

# First-visit Monte Carlo
print('First-visit Monte Carlo:')
policy = FirstVisitMC(
    env=env,
    starting_value=TabularActionValue(n_states, n_actions),
).train(n_episodes)
print_policy(policy)

# Every-visit Monte Carlo
print('Every-visit Monte Carlo:')
policy = EveryVisitMC(
    env=env,
    starting_value=TabularActionValue(n_states, n_actions),
).train(n_episodes)
print_policy(policy)

# Sarsa
print('Sarsa:')
policy = Sarsa(
    env=env,
    starting_value=TabularActionValue(n_states, n_actions),
).train(n_episodes)
print_policy(policy)

# n-step Sarsa
print('n-step Sarsa:')
policy = nStepSarsa(
    env=env,
    starting_value=TabularActionValue(n_states, n_actions),
    n=4,
).train(n_episodes)
print_policy(policy)

# Q-learning
print('Q-learning:')
policy = QLearning(
    env=env,
    starting_value=TabularActionValue(n_states, n_actions),
).train(n_episodes)
print_policy(policy)
