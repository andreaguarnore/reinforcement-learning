__all__ = [
    'FirstVisitMC',
    'EveryVisitMC',
]


from gym import Env
import numpy as np

from core.agent import MonteCarloAgent, ValueBasedAgent


class OnPolicyMCAgent(ValueBasedAgent, MonteCarloAgent):
    """
    Generic on-policy Monte Carlo agent class.
    """

    def __init__(self, env: Env, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.counts = np.zeros((self.n_states, self.n_actions), dtype=int)

    def _episode(self, n_steps: int | None = None) -> tuple[int, float]:

        # Generate an episode
        sa_pairs, rewards = self.generate_episode(
            sample_function=self.pi.sample_epsilon_greedy,
            sample_function_args={'epsilon': self.epsilon()},
            n_steps=n_steps,
        )

        # Make a list of all visited state-action pairs
        # for which the return has to be computed
        visits_to_update = self.visits_to_update(sa_pairs)

        # For all of those pairs
        for step, (state, action) in visits_to_update:

            # Compute the return starting from the step of the visit
            G = sum([reward * self.gamma ** step for step, reward in enumerate(rewards[step:])])

            # Update value
            self.counts[state, action] += 1
            update = (G - self.Q.of(state, action)) / self.counts[state, action]
            self.Q.update(state, action, update)

        return len(rewards), sum(rewards)


class FirstVisitMC(OnPolicyMCAgent):
    """
    First-visit Monte Carlo. Compute return for each first-visit to a
    state-action pair.
    """

    def visits_to_update(self, sa_pairs: list) -> list:
        first_visits = []
        for step, sa_pair in enumerate(sa_pairs):
            if sa_pair not in first_visits:
                first_visits.append((step, sa_pair))
        return first_visits


class EveryVisitMC(OnPolicyMCAgent):
    """
    Every-visit Monte Carlo. Compute return for each visit to a state-action
    pair.
    """

    def visits_to_update(self, sa_pairs: list) -> list:
        return enumerate(sa_pairs)
