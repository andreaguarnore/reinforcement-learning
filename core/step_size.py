__all__ = [
    'StepSize',
]


import numpy as np


class StepSize:
    """
    Step size class. Supports a constant, a linearly decaying, and an
    exponentially decaying step size.
    """

    def __init__(
        self,
        mode: str,
        initial_step_size: float = 0.2,
        decay: float = 1e-3,
        starting_episode: int = 0,
    ) -> None:
        assert mode in ['constant', 'linear', 'exponential'], 'Invalid step size mode'
        assert 0.0 < initial_step_size <= 1.0, 'The initial step size be in (0, 1]'
        assert decay > 0.0, 'The decay must be a positive value'
        assert starting_episode >= 0, 'The starting episode cannot be negative'
        self.initial_step_size = initial_step_size
        self.decay = decay
        self.episode = starting_episode
        self.update = getattr(type(self), mode)
        self.step_size = self.update(self)

    def __call__(self) -> float:
        return self.step_size

    def constant(self) -> float:
        return self.initial_step_size

    def linear(self) -> float:
        return self.initial_step_size / (1.0 + (self.decay * self.episode))
    
    def exponential(self) -> float:
        return self.initial_step_size * np.exp(-self.decay * self.episode)

    def next(self) -> None:
        self.episode += 1
        self.step_size = self.update(self)
