import gym
import numpy as np

from agents import *
from core.step_size import StepSize
from core.value import LinearApproxActionValue
from utils.experiment import StepsPerEpisode
from utils.featurized_states import RadialBasisFunction


n_episodes = 500
average_over = 30
gamma = 0.9
n_runs_eval = 10
evaluation = open('eval.dat', 'w')
evaluation.write('gamma features steps reward\n')

gym.envs.register(
    id='ModifiedEnv',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=10_000,
)
base_env = gym.make('ModifiedEnv', new_step_api=True)
os = base_env.observation_space
# base_env = gym.make('MountainCar-v0', new_step_api=True)
n_actions = base_env.action_space.n

rbf_gammas = [5.0, 10.0, 25.0]
all_n_features = [100, 500, 1000]
for rbf_gamma in rbf_gammas:

    for n_features in all_n_features:

        print(f'rbf gamma: {rbf_gamma}, # features: {n_features}')

        # Create environment
        env = RadialBasisFunction(
            env=base_env,
            limits=list(zip(os.low, os.high)),
            gamma=rbf_gamma,
            n_centers=n_features,
            new_step_api=True,
        )

        # Create agent
        agent = Sarsa(
            env=env,
            initial_value=LinearApproxActionValue(n_features, n_actions),
            gamma=gamma,
            alpha=StepSize('linear', 0.1, 1e-2),
            epsilon=StepSize('linear', 0.8, 1e-2),
        )

        # Run experiment
        experiment = StepsPerEpisode(env)
        training_steps, eval_steps, eval_reward = experiment.run(
            agent,
            n_episodes + average_over,
            n_runs_eval,
        )

        # Compute moving average
        training_steps = np.convolve(
            training_steps,
            np.ones(average_over),
            'valid'
        ) / average_over

        # Save steps to file
        with open(f'{int(rbf_gamma)}_{n_features}_steps.dat', 'w') as file:
            file.write('episode steps\n')
            for episode, steps in enumerate(training_steps):
                file.write(f'{episode} {steps}\n')

        # Save evaluation
        evaluation.write(f'{int(rbf_gamma)} {n_features} {eval_steps} {eval_reward}\n')

evaluation.close()
