__all__ = [
    'print_policy',
    'eval_episodes',
    'eval_training',
    'eval_learned_policy',
    'plot_errors',
    'plot_values',
]


from os.path import join

import matplotlib.pyplot as plt
import numpy as np


def print_policy(policy, states_ignored, rows=4, cols=4):
    """
    Print the learned policy for each state of the lake.
    """
    print('   policy:')
    for r in range(rows):
        print('      ', end='')
        for c in range(cols):
            state = r * cols + c
            if state not in states_ignored:
                action = policy.sample_greedy(state)
                if action == 0: action_str = '← '
                elif action == 1: action_str = '↓ '
                elif action == 2: action_str = '→ '
                else: action_str = '↑ '
                print(action_str, end='')
            else:
                print('- ', end='')
        print()


def read_episode(log_file):
    """
    Read an episode from a log file.
    """
    all_episodes = []
    episode = []
    prev_episode = 0
    with open(log_file) as file:
        for line in file:
            curr_episode, state, action, reward = line.split()
            curr_episode, state, action, reward = int(curr_episode), int(state), int(action), float(reward)
            if prev_episode != curr_episode:
                prev_episode = curr_episode
                all_episodes.append(episode)
                episode = []
            episode.append((state, action, reward))
    return all_episodes


def eval_episodes(episodes, title, generated):

    # Count steps needed to reach special state for each episode
    reached_goal = []  # steps to reach goal
    first_reach_goal = None  # number of the first episode in which the goal was reached
    reached_goal_after_first = 0  # times reached goal after having reached goal for the first time
    reached_hole = []  # steps to reach hole
    for i, e in enumerate(episodes, start=1):
        _, _, reward = e[-1]
        if reward == 1:
            reached_goal.append(len(e))
            if first_reach_goal is None:
                first_reach_goal = i
            else:
                reached_goal_after_first += 1
        else:
            reached_hole.append(len(e))

    # Print results
    print(f'   {title.format(len(episodes))}')
    print(f'      pct reached goal:                  {len(reached_goal) / len(episodes) * 100.:.2f}')
    if not generated:
        print(f'      pct reached goal after first goal: {reached_goal_after_first / max(len(episodes) - (first_reach_goal or 0.), 1.) * 100:.2f}')
    print(f'      avg steps to reach goal:           {sum(reached_goal) / max(len(reached_goal), 1.):.2f}')
    print(f'      avg steps to reach hole:           {sum(reached_hole) / max(len(reached_hole), 1.):.2f}')
    print(f'      sd of steps to reach goal:         {np.std(reached_goal):.2f}')
    print(f'      sd of steps to reach goal:         {np.std(reached_hole):.2f}')


def eval_training(method_name):
    episodes = read_episode(join('logs', f'{method_name}.log'))
    eval_episodes(episodes, 'results over {} episodes of training', False)


def eval_learned_policy(env, n_runs, policy):

    # Generate `n_runs` episodes using the learned policy
    episodes = []
    for i in range(n_runs):
        episode = []
        state = env.reset()
        while True:
            action = policy.sample_greedy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
        episodes.append(episode)

    # Evaluate the generated episodes
    eval_episodes(episodes, 'results over {} generated runs of evaluation:', True)

    return episodes


def plot_errors(methods, all_errors, train_episodes):
    nrows = 2
    ncols = 3
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)

    for i, (method, errors) in enumerate(zip(methods, all_errors)):
        row = i // ncols
        col = i % ncols
        axs[row, col].plot(np.arange(len(errors)) * train_episodes, errors)
        axs[row, col].set_xlabel('episode')
        axs[row, col].set_ylabel('mse')
        axs[row, col].set_title(method.__name__)

    # Remove unused subplots
    for i in range(len(methods), nrows * ncols):
        row = i // ncols
        col = i % ncols
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def plot_values(methods, values):
    nrows = 3
    ncols = 3
    fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)

    for i, (method, value) in enumerate(zip(methods, values)):
        row = i // ncols
        col = i % ncols
        axs[row, col].plot(np.arange(value.size), value)
        axs[row, col].set_xlabel('state')
        axs[row, col].set_ylabel('value')
        axs[row, col].set_title(method.__name__)

    # Remove unused subplots
    for i in range(len(methods), nrows * ncols):
        row = i // ncols
        col = i % ncols
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.show()
