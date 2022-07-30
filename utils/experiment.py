__all__ = [
    'Experiment',
    'MSEExperiment',
]


from copy import deepcopy

from gym import Env
import numpy as np

from agents.dynamic_programming import ValueIteration
from core.agent import Agent
from core.value import TabularStateValue


class CumulativeReward:
    """
    Cumulative reward obtained for each episode.
    """

    def __init__(self, agent: Agent, env: Env) -> None:
        self.agent = agent
        self.env = env

    def run(
        self,
        n_episodes: int,
        average_over: int,
        verbose: bool = True,
    ) -> None:

        # Compute rewards over some episodes
        rewards = np.zeros(n_episodes + average_over)
        for episode in range(n_episodes + average_over):
            if verbose:
                print(f'episode: {episode}')
            _, reward = self.agent.episode()
            rewards[episode] += reward
            episode += 1

        # Compute moving average
        rewards = np.convolve(rewards, np.ones(average_over), 'valid') / average_over

        # Save to file
        with open(f'rewards.dat', 'w') as file:
            file.write('episode reward\n')
            for episode, reward in enumerate(rewards):
                file.write(f'{episode} {reward}\n')

        return self.agent


class MeanSquaredError:
    """
    Mean squared error with different agents.
    """

    def __init__(
        self,
        env: Env,
        gamma: float = 0.9,
    ) -> None:
        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n

        # Compute optimal value
        vi = ValueIteration(
            env=env,
            starting_value=TabularStateValue(self.n_states),
            gamma=gamma,
            theta=1e-3,
        )
        vi.solve()
        self.optimal_value = vi.value.to_array()

    def run(
        self,
        agent: Agent,
        n_runs: int,
        episodes_to_log: list,
        verbose: bool = True,
    ) -> np.ndarray:
        base_agent = agent
        error = np.zeros((len(episodes_to_log), n_runs))

        # Average error over some runs
        for run in range(n_runs):

            if verbose:
                print(f'    run {run + 1}/{n_runs}')

            # Error after some episodes of training
            episode = 0
            agent = deepcopy(base_agent)
            while episode <= episodes_to_log[-1]:
                agent.train(1)
                episode += 1
                if episode in episodes_to_log:
                    value = agent.value.to_array()
                    error[episodes_to_log.index(episode), run] += np.sum(
                        np.power(self.optimal_value - value, 2)
                    ) / self.n_states

        return error
