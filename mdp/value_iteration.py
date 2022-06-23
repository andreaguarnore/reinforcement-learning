__all__ = [
    'ValueIteration',
]


from gym import Env
import numpy as np

from mdp import (
    DPMethod,
    TabularStateValue, TabularPolicy,
    one_step_lookahead,
)


class ValueIteration(DPMethod):
    """
    Policy evaluation and policy improvement: Bellman optimality backup.
    """

    def __init__(
        self,
        starting_value: TabularStateValue,
        n_states: int,
        n_actions: int,
        P: dict,
        **kwargs,
    ) -> None:
        super().__init__(n_states, n_actions, P, **kwargs)
        self.V = starting_value

    def iteration(self) -> bool:
        assert self.converged is False, 'Convergence already reached'

        delta = 0.
        for state in range(self.n_states):

            # Find maximum action-value
            action_values = one_step_lookahead(self.n_actions, self.P, state, self.V, self.gamma)
            max_action_value = np.max(action_values)

            # Update the value function
            update = max_action_value - self.V.of(state)
            delta = max(delta, abs(update))
            self.V.update(state, update)

        # If converged, compute the deterministic optimal policy
        if delta < self.theta:
            self.pi = self.policy
            self.converged = True

        return self.converged

    @property
    def value(self) -> TabularStateValue:
        return self.V

    @property
    def policy(self) -> TabularPolicy:

        # Return if already computed after convergence
        if hasattr(self, 'pi'):
            return self.pi

        # Compute intermediate policy
        policy = TabularPolicy(self.n_states, self.n_actions)
        for state in range(self.n_states):
            action_values = one_step_lookahead(self.n_actions, self.P, state, self.V, self.gamma)
            best_action = np.argmax(action_values)
            policy.make_deterministic(state, best_action)
        return policy
