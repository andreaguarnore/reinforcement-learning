__all__ = [
    'Reinforce',
    'ActorCritic',
]


from copy import deepcopy

from gym import Env
import numpy as np

from policy import ParameterizedPolicy
from rl import PolicyBasedMethod
from utils import generate_episode
from value import LinearApproxActionValue


class Reinforce(PolicyBasedMethod):
    """
    Monte Carlo policy gradient.
    """

    def train_episode(
        self,
        n_steps: int | None = None,
        save_episode: bool = False,
    ) -> None | list[tuple[int | float, int | float, float]]:

        # Generate an episode
        episode = generate_episode(
            self.env,
            self.pi.sample,
            dict(),
            n_steps,
        )

        # Compute all returns
        returns = np.zeros(len(episode))
        _, _, returns[-1] = episode[-1]
        for i, (_, _, reward) in enumerate(episode[:-1], start=2):
            returns[-i] = reward + self.gamma * returns[-i + 1]

        # Update weights at each step of the episode
        for step, ((state, action, _), G) in enumerate(zip(episode, returns)):
            update = self.alpha * self.gamma ** step * G
            self.pi.update(state, action, update)

        # Update the learning rate
        self.pi.step()

        if save_episode:
            return episode


class ActorCritic(PolicyBasedMethod):
    """
    Policy gradient with the following parameters:
        - a critic, which updates the value
        - an actor, which updates the policy in the direction suggested by the critic
    """

    def __init__(
        self,
        env: Env,
        starting_policy: ParameterizedPolicy,
        starting_value: LinearApproxActionValue,
        **kwargs,
    ) -> None:
        super().__init__(env, starting_policy, **kwargs)
        self.Q = deepcopy(starting_value)

    def train_episode(
        self,
        n_steps: int | None = None,
        save_episode: bool = False,
    ) -> None | list[tuple[int | float, int | float, float]]:

        if save_episode:
            episode = []

        # Initialize S and A
        state = self.env.reset()
        action = self.pi.sample(state)

        # For each step of the episode
        step = 0
        while n_steps is None or step < n_steps:

            # Take action A, observe R, S'
            next_state, reward, done, _ = self.env.step(action)

            if save_episode:
                episode.append((state, action, reward))

            # Choose A' from S' using the policy
            next_action = self.pi.sample(next_state)

            # Update policy parameters
            update = self.alpha * self.Q.of(state, action)
            self.pi.update(state, action, update)

            # Compute TD error
            target = reward + self.gamma * self.Q.of(next_state, next_action)
            error = target - self.Q.of(state, action)

            # Update value parameters
            update = self.alpha * error
            self.Q.update(state, action, update)

            # Stop if the environment has terminated
            if done:
                break

            # Prepare for the next step
            state = next_state
            action = next_action
            step += 1

        # Update the learning rates
        self.pi.step()
        self.Q.step()

        if save_episode:
            return episode
