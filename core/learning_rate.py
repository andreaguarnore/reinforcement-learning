__all__ = [
    'LearningRate',
]


import numpy as np


class LearningRate:

    def __init__(
        self,
        mode: str,
        lr0: float = 0.2,
        decay: float = 1e-3,
        starting_episode: int = 0
    ) -> None:
        assert mode in ['constant', 'linear', 'exponential'], 'Invalid learning rate type'
        assert 0.0 < lr0 <= 1.0, 'The starting learning rate must be in (0, 1]'
        assert decay > 0.0, 'The decay must be a positive value'
        assert starting_episode >= 0, 'The starting episode cannot be negative'
        self.episode = starting_episode
        if mode == 'constant': self.update = lambda: lr0
        elif mode == 'linear': self.update = lambda: lr0 / (1. + (decay * self.episode))
        else: self.update = lambda: lr0 * np.exp(-decay * self.episode)
        self.lr = self.update()

    def __call__(self) -> float:
        return self.lr

    def next(self) -> None:
        self.episode += 1
        self.lr = self.update()
