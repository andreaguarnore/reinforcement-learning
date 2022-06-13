import gym
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

from common import FeaturizedEnv, LinearApproxActionValue
from methods import *


# Variables of the example
max_episode_steps = 10_000  # max number of steps of the environment
n_features = 100  # number of features to be used
n_samples = 1_000  # samples used to fit the featurizer
n_tiles = 100  # precision of the animated plot
n_frames = 50  # total frames to be computed
fps = 10  # frames per second of the animation

# Linearly approximated action-value function wrapper
# which keeps track of all performed updates.
class LAAVWrapper(LinearApproxActionValue):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.wh = []  # weights history

    def update(
        self,
        features: tuple[float, ...],
        action: int,
        update: float
    ) -> None:
        super().update(features, action, update)
        self.wh.append(self.w.copy())

# Initialize default environment
gym.envs.register(
    id='MyMountainCar',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=max_episode_steps,
)
base_env = gym.make('MyMountainCar')

# Create and fit featurizer
# to the default environment
featurizer = Pipeline([
    ('scaler', StandardScaler()),
    ('rbf_sampler', RBFSampler(gamma=1., n_components=n_features)),
])
samples = np.array([base_env.observation_space.sample() for _ in range(n_samples)])
featurizer.fit(samples)

# Create wrapped environment
env = FeaturizedEnv(base_env, featurizer)
os = env.observation_space

# Train using SARSA
print('training!')
Q = LAAVWrapper(n_features, env.action_space.n)
Q, policy = sarsa(
    env,
    Q,
    epsilon=.1,
    n_episodes=1,
)

# Data for each frame of the animation
def animate(i):
    global n_frames, featurizer, Q, X, Y, surface
    episode = int(i / n_frames * len(Q.wh))
    print(f'{episode}/{len(Q.wh)}')
    Z = np.apply_along_axis(
        lambda _: -np.max(
            Q.wh[episode].T @ featurizer.transform([_]).squeeze()
            # Q.all_values(featurizer.transform([_]).squeeze())
        ),
        2,
        np.dstack([X, Y])
    )
    ax.collections[-1].remove()
    ax.plot_surface(X, Y, Z, cmap='coolwarm')
    ax.set_title(f'step {episode}')

# Initialize figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('position')
ax.set_ylabel('velocity')
ax.set_zlabel('cost to go')
ax.set_zlim(0, 12)
ax.set_title('step 0')

# Compute initial surface
X, Y = np.meshgrid(
    np.linspace(os.low[0], os.high[0], num=n_tiles),
    np.linspace(os.low[1], os.high[1], num=n_tiles),
)
surface = ax.plot_surface(X, Y, np.zeros((n_tiles, n_tiles)), cmap='coolwarm')

# Create animation and save it
print('animation!')
anim = animation.FuncAnimation(fig, animate, frames=n_frames)
anim.save('examples/fa_anim.mp4', fps=fps)
