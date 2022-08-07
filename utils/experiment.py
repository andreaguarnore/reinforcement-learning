from copy import deepcopy

from gym import Env
import numpy as np

from agents.dynamic_programming import ValueIteration
from core.agent import Agent
from core.value import TabularStateValue


class Experiment:
    """
    Generic experiment class.
    """

    def __init__(self, env: Env):
        self.env = env

    def prepare_experiment(self, n_runs: int, episodes_to_log: range) -> None:
        """
        Called before each experiment.
        """

    def log_episode(self, agent: Agent, run: int, logged_episodes: int) -> None:
        """
        Train for one episode and log results.
        """

    def log_run(self, agent: Agent, run: int) -> None:
        """
        Called after each run has ended.
        """

    def experiment_results(self):
        """
        Called once the runs have terminated in order to return all results of
        the experiment.
        """

    def run_experiment(
        self,
        agent: Agent,
        n_runs: int,
        episodes_to_log: range,
        verbosity: int = 0,
    ):
        # Prepare experiment
        self.prepare_experiment(n_runs, episodes_to_log)
        base_agent = agent
        last_episode = episodes_to_log[-1]

        # For each run
        for run in range(n_runs):

            # Prepare agent
            if verbosity > 0: print(f'run {run + 1}/{n_runs}')
            agent = deepcopy(base_agent)

            # For each episode
            episode = 0
            logged_episodes = 0
            while episode < last_episode:

                # Log episodes if needed, otherwise simply train for one episode
                episode += 1
                if episode in episodes_to_log:
                    if verbosity > 1: print(f'    episode {episode}/{last_episode + 1}')
                    self.log_episode(agent, run, logged_episodes)
                    logged_episodes += 1
                else:
                    agent.train(1)

            # Log results of the run
            self.log_run(agent, run)

        return self.experiment_results()


class CumulativeReward(Experiment):
    """
    Cumulative reward obtained for each episode over some runs.
    """

    def prepare_experiment(self, n_runs: int, episodes_to_log: range) -> None:
        self.rewards = np.zeros((len(episodes_to_log), n_runs))

    def log_episode(self, agent: Agent, run: int, logged_episodes: int) -> None:
        _, reward = agent.episode()
        self.rewards[logged_episodes, run] = reward

    def experiment_results(self):
        return self.rewards


class MeanSquaredError(Experiment):
    """
    Mean squared error (to the true value function) for each wanted episode over
    some runs.
    """

    def __init__(self, env: Env, gamma: float, **args) -> None:
        super().__init__(env, **args)
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n

        # Compute optimal value via value iteration
        vi = ValueIteration(
            env=self.env,
            initial_value=TabularStateValue(self.n_states),
            gamma=gamma,
            theta=1e-3,
        )
        vi.solve()
        self.optimal_value = vi.value.to_array()

    def prepare_experiment(self, n_runs: int, episodes_to_log: range) -> None:
        self.error = np.zeros((len(episodes_to_log), n_runs))

    def log_episode(self, agent: Agent, run: int, logged_episodes: int) -> None:
        agent.train(1)

        # Compute mean squared error
        value = agent.value.to_array()
        self.error[logged_episodes, run] = np.sum(
            np.power(self.optimal_value - value, 2)
        ) / self.n_states

    def experiment_results(self):
        return self.error


class StepsPerEpisode(Experiment):
    """
    Evaluate a trained agent by counting the number of steps and the reward
    obtained over some runs while using the implicitly derived greedy policy as
    policy.
    """

    def __init__(self, env: Env, n_eval_runs: int, **args):
        super().__init__(env, **args)
        self.n_eval_runs = n_eval_runs

    def prepare_experiment(self, n_runs: int, episodes_to_log: range) -> None:
        self.eval_steps = np.zeros(n_runs)
        self.eval_reward = np.zeros(n_runs)

    def log_episode(self, agent: Agent, run: int, logged_episodes: int) -> None:
        agent.train(1)

    def log_run(self, agent: Agent, run: int) -> None:
        policy = agent.policy
        for _ in range(self.n_eval_runs):
            state = self.env.reset()
            while True:
                action = policy.sample_greedy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.eval_steps[run] += 1.0
                self.eval_reward[run] += reward
                if terminated or truncated:
                    break
                state = next_state
        self.eval_steps[run] /= self.n_eval_runs
        self.eval_reward[run] /= self.n_eval_runs

    def experiment_results(self):
        return self.eval_steps, self.eval_reward
