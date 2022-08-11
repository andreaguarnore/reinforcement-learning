import gym

from agents import *
from core import *
from utils.featurized_states import *


n_episodes = 10_000
n_features = 1_000

# Create environment
# gym.envs.register(
#     id='ModifiedEnv',
#     entry_point='gym.envs.classic_control:MountainCarEnv',
#     max_episode_steps=10_000,
# )
base_env = gym.make('MountainCarContinuous-v0', new_step_api=True)
os = base_env.observation_space
aspace = base_env.action_space
env = RadialBasisFunction(
    env=base_env,
    limits=list(zip(os.low, os.high)),
    gamma=10.0,
    n_centers=n_features,
    new_step_api=True,
)
# n_actions = env.action_space.n

# Sarsa
# print('Sarsa:')
# policy = Sarsa(
#     env=env,
#     initial_value=LinearApproxActionValue(n_features, n_actions),
# ).train(n_episodes, verbose=True)

# # Reinforce
# print('Reinforce:')
# policy = Reinforce(
#     env=env,
#     initial_policy=SoftmaxPolicy(n_features, n_actions),
# ).train(n_episodes, verbose=True)

# # Reinforce with baseline
# print('Reinforce with baseline:')
# policy = ReinforceAdvantage(
#     env=env,
#     initial_policy=SoftmaxPolicy(n_features, n_actions),
#     initial_value=LinearApproxStateValue(n_features),
# ).train(n_episodes, verbose=True)

# # Actor-critic
print('Actor-critic:')
policy = ActorCritic(
    env=env,
    # initial_policy=SoftmaxPolicy(n_features, n_actions),
    initial_policy=GaussianPolicy(n_features, std_dev=1.0),
    initial_value=LinearApproxStateValue(n_features),
).train(n_episodes, verbose=True)
