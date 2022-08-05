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
rbf_gamma = 10
n_features = 500
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
env = RadialBasisFunction(
    env=base_env,
    limits=list(zip(os.low, os.high)),
    gamma=rbf_gamma,
    n_centers=n_features,
    new_step_api=True,
)
n_actions = base_env.action_space.n

lambdas = [0.1, 0.3, 0.6, 0.9, 0.95, 0.975, 0.99]
for lambda_ in lambdas:

    print(lambda_)

    # Create agent
    agent = SarsaLambda(
        env=env,
        initial_value=LinearApproxActionValue(n_features, n_actions),
        gamma=gamma,
        alpha=StepSize('linear', 0.1, 1e-2),
        epsilon=StepSize('linear', 0.8, 1e-2),
        lambda_=lambda_,
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
    with open(f'{lambda_}.dat', 'w') as file:
        file.write('episode steps\n')
        for episode, steps in enumerate(training_steps):
            file.write(f'{episode} {steps}\n')

    # Save evaluation
    evaluation.write(f'{int(lambda_) if lambda_ == 0 else lambda_} {eval_steps} {eval_reward}\n')

evaluation.close()
