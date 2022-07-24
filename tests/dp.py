import gym

from agents import PolicyIteration, ValueIteration
from core import TabularPolicy, TabularStateValue


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

# Create environment
env = gym.make('FrozenLake-v1', is_slippery=True, new_step_api=True)
n_states = env.observation_space.n
n_actions = env.action_space.n

# Policy iteration
policy = PolicyIteration(
    env=env,
    starting_policy=TabularPolicy(n_states, n_actions),
    gamma=0.9,
    theta=1e-3,
).solve()
print('Policy iteration:')
print_policy(policy)

# Value iteration
policy = ValueIteration(
    env=env,
    starting_value=TabularStateValue(n_states),
    gamma=0.9,
    theta=1e-3,
).solve()
print('Value iteration:')
print_policy(policy)
