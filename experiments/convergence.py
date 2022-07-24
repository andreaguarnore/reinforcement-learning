import gym

from agents import *
from core.agent import Agent
from core.learning_rate import LearningRate
from core.value import TabularActionValue
from utils.experiment import Convergence


def agent_generator() -> Agent:
    while True:
        yield Sarsa(
            env=env,
            starting_value=TabularActionValue(n_states, n_actions),
            gamma=gamma,
            epsilon=LearningRate('linear', 0.8, 1e-3),
        )


env = gym.make('FrozenLake-v1', is_slippery=True)
n_states = env.observation_space.n
n_actions = env.action_space.n
holes = [5, 7, 11, 12, 15]
gamma = 0.9
experiment = Convergence(agent_generator, env, holes, gamma)
experiment.run()
