import gym

from agents import *
from core import *
from utils.featurized_states import *


n_episodes = 9_000
n_features = 10_000

# Create environment
gym.envs.register(
    id='ModifiedEnv',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=10_000,
)
base_env = gym.make('ModifiedEnv', new_step_api=True)
env = RadialBasisFunction(
    env=base_env,
    standardize=True,
    n_samples=1_000,
    gamma=10.0,
    n_features=n_features,
    new_step_api=True,
)
n_actions = env.action_space.n

# Sarsa
print('Sarsa:')
policy = Sarsa(
    env=env,
    starting_value=LinearApproxActionValue(n_features, n_actions),
).train(n_episodes, verbose=True)

# # Reinforce
# print('Reinforce:')
# policy = Reinforce(
#     env=env,
#     starting_policy=SoftmaxPolicy(n_features, n_actions),
# ).train(n_episodes, verbose=True)

# # Reinforce with baseline
# print('Reinforce with baseline:')
# policy = ReinforceAdvantage(
#     env=env,
#     starting_policy=SoftmaxPolicy(n_features, n_actions),
#     starting_value=LinearApproxStateValue(n_features),
# ).train(n_episodes, verbose=True)

# # Actor-critic
# print('Actor-critic:')
# policy = ActorCritic(
#     env=env,
#     starting_policy=SoftmaxPolicy(n_features, n_actions),
#     starting_value=LinearApproxStateValue(n_features),
# ).train(n_episodes, verbose=True)
