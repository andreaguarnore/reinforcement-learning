__all__ = [
    'LearningRate',
]


import numpy as np


class LearningRate:
    """
    Learning rate class. Supports a constant, a linearly decaying, and an
    exponentially decaying learning rate.
    """

    def __init__(
        self,
        mode: str,
        lr0: float = 0.2,
        decay: float = 1e-3,
        starting_episode: int = 0,
    ) -> None:
        assert mode in ['constant', 'linear', 'exponential'], 'Invalid learning rate type'
        assert 0.0 < lr0 <= 1.0, 'The starting learning rate must be in (0, 1]'
        assert decay > 0.0, 'The decay must be a positive value'
        assert starting_episode >= 0, 'The starting episode cannot be negative'
        self.lr0 = lr0
        self.decay = decay
        self.episode = starting_episode
        self.update = getattr(type(self), mode)
        self.lr = self.update(self)

    def __call__(self) -> float:
        return self.lr

    def constant(self) -> float:
        return self.lr0

    def linear(self) -> float:
        return self.lr0 / (1.0 + (self.decay * self.episode))
    
    def exponential(self) -> float:
        return self.lr0 * np.exp(-self.decay * self.episode)

    def next(self) -> None:
        self.episode += 1
        self.lr = self.update(self)
