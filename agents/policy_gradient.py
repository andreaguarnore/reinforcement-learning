__all__ = [
    'Reinforce',
    'ReinforceAdvantage',
    'ActorCritic',
]


from gym import Env

from core.agent import MonteCarloAgent, PolicyBasedAgent
from core.step_size import StepSize
from core.value import LinearApproxStateValue


class Reinforce(PolicyBasedAgent, MonteCarloAgent):
    """
    Monte Carlo policy gradient.
    """

    def _episode(self, n_steps: int | None = None) -> tuple[int, float]:

        # Sample an episode
        sa_pairs, rewards = self.generate_episode(
            sample_function=self.pi.sample,
            sample_function_args=dict(),
            n_steps=n_steps,
        )

        # For all state-action pairs
        for step, (state, action) in enumerate(sa_pairs):

            # Compute the return starting from the step of the visit
            G = sum([reward * self.gamma ** step for step, reward in enumerate(rewards[step:])])

            # Update policy
            self.pi.update(state, action, self.alpha() * G)

        return len(rewards), sum(rewards)


class ReinforceAdvantage(PolicyBasedAgent, MonteCarloAgent):
    """
    Monte Carlo policy gradient using an approximate state-value function as
    baseline.
    """

    def __init__(
        self,
        env: Env,
        initial_value: LinearApproxStateValue,
        value_alpha: StepSize = None,
        **kwargs
    ) -> None:
        super().__init__(env, **kwargs)
        self.V = initial_value
        self.value_alpha = StepSize('linear') if value_alpha is None else value_alpha

    def _episode(self, n_steps: int | None = None) -> tuple[int, float]:

        # Sample an episode
        sa_pairs, rewards = self.generate_episode(
            sample_function=self.pi.sample,
            sample_function_args=dict(),
            n_steps=n_steps,
        )

        # For all state-action pairs
        for step, (state, action) in enumerate(sa_pairs):

            # Compute the advantage function
            G = sum([reward * self.gamma ** step for step, reward in enumerate(rewards[step:])])
            A = G - self.V.of(state)

            # Update value and policy
            self.V.update(state, self.value_alpha() * A)
            self.pi.update(state, action, self.policy_alpha() * G)

        return len(rewards), sum(rewards)

    @property
    def policy_alpha(self):
        return self.alpha


class ActorCritic(PolicyBasedAgent):
    """
    One-step actor-critic.
    """

    def __init__(
        self,
        env: Env,
        initial_value: LinearApproxStateValue,
        value_alpha: StepSize = None,
        **kwargs
    ) -> None:
        super().__init__(env, **kwargs)
        self.V = initial_value
        self.value_alpha = StepSize('linear') if value_alpha is None else value_alpha

    def _episode(self, n_steps: int | None = None) -> tuple[int, float]:

        # Sample the starting state
        state = self.env.reset()

        # For each step of the episode
        step = 0
        total_reward = 0.
        while n_steps is None or step < n_steps:

            # Sample an action from the current policy
            action = self.pi.sample(state)

            # Sample the next state and
            # the reward associated with the last transition
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            step += 1
            total_reward += reward

            # Compute TD error
            target = reward + self.gamma * self.V.of(next_state)
            error = target - self.V.of(state)

            # Update value and policy
            self.V.update(state, self.value_alpha() * error)
            self.pi.update(state, action, self.policy_alpha() * error)

            # Stop if the environment has terminated
            if terminated or truncated:
                break

            # Prepare for the next step
            state = next_state

        return step, total_reward

    @property
    def policy_alpha(self):
        return self.alpha
