__all__ = [
    'Experiment',
    'MSEExperiment',
]


from gym import Env
import numpy as np

from agents.dynamic_programming import PolicyIteration
from core.agent import Agent
from core.policy import TabularPolicy


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
    ) -> None:

        # Compute rewards over some episodes
        rewards = np.zeros(n_episodes + average_over)
        for episode in range(n_episodes + average_over):
            print(episode)
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
        agent_generator: callable,
        env: Env,
        gamma: float = 0.9,
    ) -> None:
        self.agent_generator = agent_generator

        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n

        # Compute optimal value
        pi = PolicyIteration(
            env=env,
            starting_policy=TabularPolicy(self.n_states, self.n_actions),
            gamma=gamma,
            theta=1e-3,
        )
        pi.solve()
        self.optimal_value = pi.value.to_array()

    def run(
        self,
        n_runs: int,
        episodes_to_log: list,
        verbose: bool = True,
    ) -> None:

        # For each agent
        for name, agent in self.agent_generator():

            if verbose:
                print(name)

            # Average error over some runs
            error = np.zeros((len(episodes_to_log), n_runs))
            for run in range(n_runs):

                if verbose:
                    print(f' run {run + 1}/{n_runs}')

                # Error after some episodes of training
                episode = 0
                while episode <= episodes_to_log[-1]:
                    if episode in episodes_to_log:
                        value = agent.value.to_array()
                        error[episodes_to_log.index(episode), run] += np.sum(
                            np.power(self.optimal_value - value, 2)
                        ) / self.n_states
                    agent.train(1)
                    episode += 1

            # Save error
            with open(f'{name}.dat', 'w') as file:
                file.write('episode mse top bottom\n')
                for episode, mse in zip(episodes_to_log, error):
                    mean = np.mean(mse)
                    std = np.std(mse)
                    file.write(f'{episode} {mean} {mean + std} {mean - std}\n')


class Convergence:
    """
    Training until convergence to optimal policy.
    """

    def __init__(
        self,
        agent_generator: callable,
        env: Env,
        ignored_states: list | None = None,
        gamma: float = 0.9,
    ) -> None:
        self.agent_generator = agent_generator
        self.ignored_states = ignored_states if ignored_states is not None else []
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n

        # Compute optimal policy
        self.optimal_policy = PolicyIteration(
            env=env,
            starting_policy=TabularPolicy(self.n_states, self.n_actions),
            gamma=gamma,
            theta=1e-3,
        ).solve()

    def run(
        self,
        n_runs: int,
        n_episodes: int,
        episodes_log: int,
        verbose: bool = True,
    ) -> None:

        # Average convergence to optimal policy over some runs
        eps_to_opt = []
        for run in range(n_runs):

            if verbose:
                print(f'run {run + 1}/{n_runs}')

            # Error after some episodes of training
            episode = 0
            agent = next(self.agent_generator())
            while episode < n_episodes:
                policy = agent.train(episodes_log)
                for s in range(self.n_states):
                    if s not in self.ignored_states and \
                        self.optimal_policy.sample_greedy(s) != policy.sample_greedy(s):
                        break
                else:
                    eps_to_opt.append(episode)
                    break
                episode += episodes_log

        print(len(eps_to_opt))
        print(eps_to_opt)
