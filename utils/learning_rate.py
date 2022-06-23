__all__ = [
    'LearningRate',
    'ConstantLR',
    'LinearLR',
    'ExpLR',
]


import numpy as np


class LearningRate:
    """
    Learning rate determining some step size. All learning rates are initialized from episode 1.
    """

    def update(self, episode: int) -> None:
        """
        Update learning rate to the value at the given episode.
        """
        raise NotImplementedError


class ConstantLR(LearningRate):
    """
    Constant learning rate.
    """

    def __init__(self, lr: float) -> None:
        self.lr = lr

    def update(self, episode: int) -> None:
        pass


class LinearLR(LearningRate):
    """
    Linearly decaying learning rate.
    """

    def __init__(self, lr0: float = .1, decay: float = 1e-3) -> None:
        assert decay > 0., 'The decay must be a positive value'
        self.lr0 = lr0
        self.decay = decay
        self.update(1)

    def update(self, episode: int) -> None:
        self.lr = self.lr0 / (1. + (self.decay * episode))


class ExpLR(LearningRate):
    """
    Exponentially decaying learning rate.
    """

    def __init__(self, lr0: float = .8, decay: float = 1e-3) -> None:
        assert decay > 0., 'The decay must be a positive value'
        self.lr0 = lr0
        self.decay = decay
        self.update(1)

    def update(self, episode: int) -> None:
        self.lr = self.lr0 * np.exp(-self.decay * episode)
