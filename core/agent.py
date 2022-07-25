__all__ = [
    'Agent',
    'DPAgent',
    'MonteCarloAgent',
    'ValueBasedAgent',
    'PolicyBasedAgent',
]


from gym import Env
import numpy as np

from core.learning_rate import LearningRate
from core.policy import TabularPolicy, ImplicitPolicy, ParameterizedPolicy
from core.value import TabularStateValue, ActionValue, LinearApproxActionValue


class Agent:
    """
    Generic agent class.
    """

    def __init__(
        self,
        env: Env,
        gamma: float = 0.9,
    ) -> None:
        assert 0.0 <= gamma <= 1.0, 'The discount factor gamma must in [0, 1]'
        self.env = env
        self.gamma = gamma

    @property
    def value(self) -> TabularStateValue:
        """
        Returns the current value.
        """
        raise RuntimeError('This agent does not have a value')

    @property
    def policy(self) -> TabularPolicy:
        """
        Returns the current policy
        """


class DPAgent(Agent):
    """
    Generic dynamic programming agent class.
    """

    def __init__(
        self,
        env: Env,
        theta: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        assert theta > 0.0, 'The threshold theta must be positive'
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.P = env.P
        self.theta = theta
        self.converged = False

    def assert_not_converged(self) -> None:
        assert not self.converged, 'Convergence already reached'

    def one_step_lookahead(self, state: int) -> np.ndarray:
        """
        Compute all action values in a given state.
        """
        return np.array([
            np.sum([
                trans_prob * (reward + self.gamma * self.V.of(next_state)) for
                trans_prob, next_state, reward, _ in self.P[state][action]
            ]) for action in range(self.n_actions)
        ])

    def iteration(self) -> None:
        """
        A policy evaluation and a policy improvement step.
        """

    def solve(self) -> TabularPolicy:
        """
        Iterate until convergence.
        """
        self.assert_not_converged()
        while not self.converged:
            self.iteration()
        return self.pi


class RLAgent(Agent):
    """
    Generic reinforcement learning class.
    """

    def __init__(
        self,
        env: Env,
        alpha: LearningRate = None,
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        self.alpha = LearningRate(
            mode='linear',
            lr0=0.1,
            decay=1e-3,
        ) if alpha is None else alpha
        self.lrs = [self.alpha]
        self.total_episodes = 0

    def episode(
        self,
        n_steps: int | None = None,
        verbose: bool = False,
    ) -> tuple[int, float]:
        """
        Train for one episode. Returns the total number of steps and the
        cumulative reward obtained in the episode.
        """
        assert n_steps is None or n_steps > 0, 'The number of steps must be positive'

        if verbose:
            print(f'Episode: {self.total_episodes + 1}')

        # Train for one episode
        episode_steps, total_reward = self._episode(n_steps)
        self.total_episodes += 1

        # Update learning rates
        for lr in self.lrs:
            lr.next()

        if verbose:
            print(f'    Steps:  {episode_steps}')
            print(f'    Reward: {total_reward}')

        return episode_steps, total_reward

    def train(
        self,
        n_episodes: int,
        **kwargs,
    ) -> ImplicitPolicy | ParameterizedPolicy:
        """
        Train the agent for `n_episodes` episodes. Returns the learned policy.
        """
        assert n_episodes > 0, 'The number of episodes must be positive'
        for _ in range(n_episodes):
            self.episode(**kwargs)
        return self.pi

    @property
    def policy(self) -> ImplicitPolicy | ParameterizedPolicy:
        return self.pi


class MonteCarloAgent(RLAgent):
    """
    Generic Monte Carlo (can be either value- or policy-based) agent class.
    """

    def generate_episode(
        self,
        sample_function: callable,
        sample_function_args: dict,
        n_steps: int | None,
    ):
        """
        Generate an episode using the given sample function of the current
        policy.
        """
        sa_pairs = []
        rewards = []
        state = self.env.reset()
        step = 0
        while n_steps is None or step < n_steps:
            action = sample_function(state, **sample_function_args)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            sa_pairs.append((state, action))
            rewards.append(reward)
            if terminated or truncated:
                break
            state = next_state
            step += 1
        return sa_pairs, rewards


class ValueBasedAgent(RLAgent):
    """
    Generic value-based agent class.
    """

    def __init__(
        self,
        env: Env,
        starting_value: ActionValue,
        epsilon: LearningRate = None,
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        self.Q = starting_value
        if isinstance(self.Q, LinearApproxActionValue):
            self.lrs.append(self.Q.lr)
        self.pi = ImplicitPolicy(self.Q)
        self.epsilon = LearningRate(
            mode='linear',
            lr0=0.8,
            decay=1e-3,
        ) if epsilon is None else epsilon
        self.lrs.append(self.epsilon)

    @property
    def value(self) -> ActionValue:
        return self.Q


class PolicyBasedAgent(RLAgent):
    """
    Generic Policy-based agent class.
    """

    def __init__(
        self,
        env: Env,
        starting_policy: ParameterizedPolicy,
        **kwargs
    ) -> None:
        super().__init__(env, **kwargs)
        self.pi = starting_policy
        self.lrs.append(self.pi.lr)
