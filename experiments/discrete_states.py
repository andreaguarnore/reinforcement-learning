import gym
import numpy as np

from agents import *
from core.step_size import StepSize
from core.value import TabularActionValue
from utils.discretized_states import DiscretizedStates
from utils.experiment import StepsPerEpisode


n_episodes = 500
gamma = 0.9
n_runs = 10
n_eval_runs = 100
filename = 'eval_discrete.dat'
with open(filename, 'w') as file:
    file.write('bins avg_steps least_steps most_steps\n')

gym.envs.register(
    id='ModifiedEnv',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=10_000,
)
base_env = gym.make('ModifiedEnv', new_step_api=True)
n_actions = base_env.action_space.n
dims = base_env.observation_space.shape[0]

all_n_bins = [5, 10, 20, 40]
for n_bins in all_n_bins:

    print(n_bins)

    # Create discretized environment
    env = DiscretizedStates(
        env=base_env,
        n_bins=n_bins,
        new_step_api=True,
    )

    # Create agent
    agent = Sarsa(
        env=env,
        initial_value=TabularActionValue(n_bins ** dims, n_actions),
        gamma=gamma,
        alpha=StepSize('linear', 0.1, 1e-2),
        epsilon=StepSize('linear', 0.8, 1e-2),
    )

    # Run experiment
    experiment = StepsPerEpisode(env, n_eval_runs)
    eval_steps, _ = experiment.run_experiment(
        agent=agent,
        episodes_to_log=range(1, n_episodes),
        n_runs=n_runs,
        verbosity=1,
    )

    # Save evaluation
    with open(filename, 'a') as file:
        file.write(f'{n_bins} {np.mean(eval_steps)} {np.min(eval_steps)} {np.max(eval_steps)}\n')
