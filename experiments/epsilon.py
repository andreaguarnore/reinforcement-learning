import gym

from agents import *
from core.agent import Agent
from core.learning_rate import LearningRate
from core.value import TabularActionValue
from utils.experiment import MeanSquaredError


def epsilon_experiment() -> tuple[str, Agent]:
    epsilons = {
        'constant_0_2':  LearningRate('constant', 0.2),
        'linear_1e-2':   LearningRate('linear',   0.8, 1e-2),
        'linear_1e-3':   LearningRate('linear',   0.8, 1e-3),
        'linear_1e-4':   LearningRate('linear',   0.8, 1e-4),
    }
    for name, epsilon in epsilons.items():
        yield name, QLearning(
            env=env,
            starting_value=TabularActionValue(n_states, n_actions),
            gamma=gamma,
            epsilon=epsilon,
        )


env = gym.make('FrozenLake-v1', is_slippery=True)
n_states = env.observation_space.n
n_actions = env.action_space.n
gamma = 0.9
experiment = MeanSquaredError(epsilon_experiment, env, gamma)
experiment.run(
    n_runs=10,
    episodes_to_log=list(range(1, 10_000, 50)),
)
