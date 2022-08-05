import gym
import numpy as np

from agents import *
from core.step_size import StepSize
from core.value import TabularActionValue
from utils.discretized_states import DiscretizedStates
from utils.experiment import StepsPerEpisode


# gym.envs.register(
#     id='ModifiedEnv',
#     entry_point='gym.envs.classic_control:MountainCarEnv',
#     max_episode_steps=10_000,
# )
# base_env = gym.make('ModifiedEnv', new_step_api=True)
base_env = gym.make('MountainCar-v0', new_step_api=True)
n_actions = base_env.action_space.n
env = DiscretizedStates(
    env=base_env,
    n_bins=n_bins,
    new_step_api=True,
)

n_episodes = 500
average_over = 30
gamma = 0.9
alpha = StepSize('linear', 0.1, 1e-2)
epsilon = StepSize('linear', 0.8, 1e-2)

methods = [
    Sarsa,
    QLearning,
    Reinforce,
    ReinforceBaseline,
    ActorCritic,
]
for method in methods:

    print(method.__name__)

    # Create discretized environment
    

    # Create agent
    agent = Sarsa(
        env=env,
        initial_value=TabularActionValue(n_bins ** dims, n_actions),
        gamma=gamma,
        alpha=alpha,
        epsilon=epsilon,
    )

    # Run experiment
    experiment = StepsPerEpisode(env)
    steps = experiment.run(
        agent,
        n_episodes + average_over,
    )

    # Compute moving average
    steps = np.convolve(steps, np.ones(average_over), 'valid') / average_over

    # Save steps to file
    with open(f'{n_bins}_steps.dat', 'w') as file:
        file.write('episode steps\n')
        for episode, steps_in_episode in enumerate(steps):
            file.write(f'{episode} {steps_in_episode}\n')

    # Save cost to go
    value = agent.value
    with open(f'{n_bins}_cost.dat', 'w') as file:
        file.write('position velocity cost\n')
        last_position = None
        for idx in range(n_bins ** dims):
            position, velocity = env.unravel_idx(idx)
            if position != last_position:
                file.write('\n')
                last_position = position
            cost_to_go = -max(value.all_values(idx))
            file.write(f'{position} {velocity} {cost_to_go}\n')
