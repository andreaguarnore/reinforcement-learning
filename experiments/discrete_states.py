import gym
import numpy as np

from agents import *
from core.step_size import StepSize
from core.value import TabularActionValue
from utils.discretized_states import DiscretizedStates
from utils.experiment import StepsPerEpisode


gym.envs.register(
    id='ModifiedEnv',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=10_000,
)
base_env = gym.make('ModifiedEnv', new_step_api=True)
n_actions = base_env.action_space.n
dims = base_env.observation_space.shape[0]

n_episodes = 500
gamma = 0.9
n_runs_eval = 10
average_over = 30
evaluation = open('eval.dat', 'w')
evaluation.write('bins steps reward\n')

all_n_bins = [5, 10, 20]
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
    experiment = StepsPerEpisode(env)
    training_steps, eval_steps, eval_reward = experiment.run(
        agent,
        n_episodes + average_over,
        n_runs_eval,
    )

    # Compute moving average
    training_steps = np.convolve(
        training_steps,
        np.ones(average_over),
        'valid',
    ) / average_over

    # Save steps to file
    with open(f'{n_bins}.dat', 'w') as file:
        file.write('episode steps\n')
        for episode, steps in enumerate(training_steps):
            file.write(f'{episode} {steps}\n')

    # Save evaluation
    evaluation.write(f'{n_bins} {eval_steps} {eval_reward}\n')

evaluation.close()
