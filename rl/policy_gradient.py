__all__ = [
    'Reinforce',
    'ActorCritic',
]


from gym import Env
import numpy as np

from policy import ParameterizedPolicy
from rl import PolicyBasedMethod
from utils import generate_episode, LinearLR
from value import LinearApproxActionValue


class Reinforce(PolicyBasedMethod):
    """
    Monte Carlo policy gradient.
    """

    def train_episode(self, n_steps: int | None = None) -> tuple[int, float]:

        # Generate an episode
        episode = generate_episode(
            self.env,
            self.pi.sample,
            dict(),
            n_steps,
        )
        if self.save_episodes:
            self.file_logger.save_episode(episode)

        # Compute all returns
        returns = np.zeros(len(episode))
        _, _, returns[-1] = episode[-1]
        for i, (_, _, reward) in enumerate(episode[:-1], start=2):
            returns[-i] = reward + self.gamma * returns[-i + 1]

        # Update weights at each step of the episode
        for step, ((state, action, _), G) in enumerate(zip(episode, returns)):
            update = self.alpha.lr * self.gamma ** step * G
            self.pi.update(state, action, update)

        # Update the learning rate
        self.pi.step()

        return len(episode), sum([r for _, _, r in episode])


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
        value_alpha: LinearLR = None,
        **kwargs,
    ) -> None:
        super().__init__(env, starting_policy, **kwargs)
        self.Q = starting_value
        self.value_alpha = LinearLR() if value_alpha is None else value_alpha
        self.lrs.append([self.value_alpha, self.Q.lr])

    def train_episode(self, n_steps: int | None = None) -> tuple[int, float]:

        # Initialize S and A
        state = self.env.reset()

        # For each step of the episode
        step = 0
        total_reward = 0.
        while n_steps is None or step < n_steps:

            # Choose A from the current policy
            action = self.pi.sample(state)

            # Take action A, observe R, S'
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            if self.save_episodes:
                self.file_logger.save_episode_step(state, action, reward)

            # Choose A' from S' using the policy
            next_action = self.pi.sample(next_state)

            # Update policy parameters
            update = self.alpha.lr * self.Q.of(state, action)
            self.pi.update(state, action, update)

            # Compute TD error
            target = reward + self.gamma * self.Q.of(next_state, next_action)
            error = target - self.Q.of(state, action)

            # Update value parameters
            update = self.value_alpha.lr * error
            self.Q.update(state, action, update)

            # Stop if the environment has terminated
            if done:
                break

            # Prepare for the next step
            state = next_state
            step += 1

        # Update the learning rates
        self.pi.step()
        self.Q.step()

        return step, total_reward

    def train(self, n_episodes: int, **kwargs) -> None:
        super().train(n_episodes, **kwargs)
        return self.Q, self.pi
