__all__ = [
    'DPMethod',
]


from gym import Env

from mdp import TabularStateValue, TabularPolicy


class DPMethod:
    """
    Generic dynamic programming method class
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        P: dict,
        gamma: float = .9,
        theta: float = 1e-3,
    ) -> None:
        assert n_states > 0, 'The number of states must be positive'
        assert n_actions > 0, 'The number of actions must be positive'
        assert 0. <= gamma <= 1., 'The discount factor gamma must in [0, 1]'
        assert theta > 0., 'The threshold theta must be positive'

        self.n_states = n_states
        self.n_actions = n_actions
        self.P = P
        self.gamma = gamma
        self.theta = theta
        self.converged = False

    def iteration(self) -> bool:
        """
        A policy evaluation and a policy improvement step.
        """
        raise NotImplementedError

    def solve(self) -> tuple[TabularStateValue, TabularPolicy]:
        """
        Iterate until convergence.
        """
        assert self.converged is False, 'Convergence already reached'
        while not self.converged:
            self.iteration()
        return self.V, self.pi

    @property
    def value(self) -> TabularStateValue:
        raise NotImplementedError

    @property
    def policy(self) -> TabularPolicy:
        raise NotImplementedError
