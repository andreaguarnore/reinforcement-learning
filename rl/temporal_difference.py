__all__ = [
    'Sarsa',
    'QLearning',
]


from gym import Env

from rl import ValueBasedMethod
from value import ActionValue, LinearApproxActionValue


class TDMethod(ValueBasedMethod):
    """
    Generic temporal difference learning method.
    """

    def __init__(
        self,
        greedy_next_action: bool,
        env: Env,
        starting_value: ActionValue,
        **kwargs,
    ):
        super().__init__(env, starting_value, **kwargs)
        self.greedy_next_action = greedy_next_action

    def train_episode(self, n_steps: int | None = None) -> tuple[int, float]:

        # Initialize S
        state = self.env.reset()

        # For each step of the episode
        step = 0
        total_reward = 0.
        while n_steps is None or step < n_steps:

            # Choose A from S epsilon greedily using the policy derived from Q
            action = self.pi.sample_epsilon_greedy(state, self.epsilon)

            # Take action A, observe R, S'
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            if self.save_episodes:
                self.file_logger.save_episode_step(state, action, reward)

            # Choose A' from S' using the behavior policy
            if self.greedy_next_action: next_action = self.pi.sample_greedy(next_state)
            else: next_action = self.pi.sample_epsilon_greedy(next_state, self.epsilon)

            # TD update
            target = reward + self.gamma * self.Q.of(next_state, next_action)
            error = target - self.Q.of(state, action)
            update = self.alpha * error
            self.Q.update(state, action, update)

            # Stop if the environment has terminated
            if done:
                break

            # Prepare for the next step
            state = next_state
            step += 1

        # Update the learning rate of the action-value function if needed
        if isinstance(self.Q, LinearApproxActionValue):
            self.Q.step()

        return step, total_reward


class Sarsa(TDMethod):
    """
    On-policy temporal difference learning.
    """

    def __init__(
        self,
        env: Env,
        starting_value: ActionValue,
        **kwargs,
    ) -> None:
        super().__init__(
            greedy_next_action=False,
            starting_value=starting_value,
            env=env,
            **kwargs,
        )


class QLearning(TDMethod):
    """
    Off-policy temporal difference learning.
    """

    def __init__(
        self,
        env: Env,
        starting_value: ActionValue,
        **kwargs,
    ) -> None:
        super().__init__(
            greedy_next_action=True,
            starting_value=starting_value,
            env=env,
            **kwargs,
        )
