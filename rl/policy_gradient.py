__all__ = [
    # 'FirstVisitMC',
    # 'OffPolicyMC',
]


# from copy import deepcopy

# from gym import Env

# from policy import DerivedPolicy
# from rl import Method
# from value import ActionValue, LinearApproxActionValue


# class TDMethod(Method):
#     """
#     Generic temporal difference learning method.
#     """

#     def __init__(
#         self,
#         greedy_next_action: bool,
#         starting_value: ActionValue,
#         env: Env,
#         gamma: float = .9,
#         epsilon: float = .3,
#         epsilon_decay: float = .1,
#         epsilon_mode: str = 'exponential',
#         alpha: float = .1,
#         alpha_decay: float = 1e-3,
#     ):
#         super().__init__(env, gamma)
#         self.greedy_next_action = greedy_next_action
#         self.Q = deepcopy(starting_value)
#         self.pi = DerivedPolicy(self.Q)
#         self.lrs.update({
#             'epsilon': (epsilon, epsilon_decay, epsilon_mode),
#             'alpha': (alpha, alpha_decay, 'robbins_monro'),
#         })

#     def train_episode(
#         self,
#         n_steps: int | None = None,
#     ) -> None:

#         # Initialize S
#         state = self.env.reset()

#         # For each step of the episode
#         step = 0
#         while n_steps is None or step < n_steps:

#             # Choose A from S epsilon greedily using the policy derived from Q
#             action = self.pi.sample_epsilon_greedy(state, self.epsilon)

#             # Take action A, observe R, S'
#             next_state, reward, done, _ = self.env.step(action)

#             # Choose A' from S' using the behavior policy
#             if self.greedy_next_action: next_action = self.pi.sample_greedy(next_state)
#             else: next_action = self.pi.sample_epsilon_greedy(next_state, self.epsilon)

#             # TD update
#             target = reward + self.gamma * self.Q.of(next_state, next_action)
#             error = target - self.Q.of(state, action)
#             update = self.alpha * error
#             self.Q.update(state, action, update)

#             # Stop if the environment has terminated
#             if done:
#                 break

#             # Prepare for the next step
#             state = next_state
#             step += 1

#         # Update the learning rate of the action-value function if needed
#         if isinstance(self.Q, LinearApproxActionValue):
#             self.Q.step()


# class Sarsa(TDMethod):
#     """
#     On-policy temporal difference learning.
#     """

#     def __init__(
#         self,
#         starting_value: ActionValue,
#         env: Env,
#         **kwargs,
#     ):
#         super().__init__(
#             greedy_next_action=False,
#             starting_value=starting_value,
#             env=env,
#             **kwargs,
#         )


# class QLearning(TDMethod):
#     """
#     Off-policy temporal difference learning.
#     """

#     def __init__(
#         self,
#         starting_value: ActionValue,
#         env: Env,
#         **kwargs,
#     ):
#         super().__init__(
#             greedy_next_action=True,
#             starting_value=starting_value,
#             env=env,
#             **kwargs,
#         )
