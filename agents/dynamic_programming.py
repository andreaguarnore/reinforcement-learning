__all__ = [
    'PolicyIteration',
    'ValueIteration',
]


from gym import Env
import numpy as np

from core.agent import DPAgent
from core.value import TabularStateValue
from core.policy import TabularPolicy


class PolicyIteration(DPAgent):
    """
    Policy evaluation: Bellman expectation backup.
    Policy improvement: Greedy policy improvement.
    """

    def __init__(
        self,
        env: Env,
        initial_policy: TabularPolicy,
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        self.pi = initial_policy

    def iteration(self) -> None:
        self.assert_not_converged()
        self.iterative_policy_evaluation()
        self.policy_improvement()

    def iterative_policy_evaluation(self) -> TabularStateValue:
        """
        Return value estimated according to the current policy.
        """
        self.V = TabularStateValue(self.n_states)

        # Until convergence
        while True:

            # For each state
            delta = 0.0
            for state in range(self.n_states):

                # Compute the value of this state
                state_value = self.pi.probabilities(state) @ self.one_step_lookahead(state)

                # Update value function
                update = state_value - self.V.of(state)
                delta = max(delta, abs(update))
                self.V.update(state, update)

            # Stop evaluation when the estimation is accurate enough
            if delta < self.theta:
                return

    def policy_improvement(self) -> None:
        """
        Improve policy towards optimality.
        """
        is_policy_optimal = True

        # For each state
        for state in range(self.n_states):

            # Find the best action under the current policy and
            # under the current state-value function
            old_best_action = self.pi.sample_greedy(state)
            new_best_action = np.argmax(self.one_step_lookahead(state))

            # Update policy
            if new_best_action != old_best_action:
                is_policy_optimal = False
            self.pi.make_deterministic(state, new_best_action)

        # Check whether convergence has been reached
        if is_policy_optimal:
            self.converged = True

    @property
    def value(self) -> TabularStateValue:
        if not self.converged:
            self.iterative_policy_evaluation()
        return self.V

    @property
    def policy(self) -> TabularPolicy:
        return self.pi


class ValueIteration(DPAgent):
    """
    Policy evaluation and policy improvement: Bellman optimality backup.
    """

    def __init__(
        self,
        env: Env,
        initial_value: TabularStateValue,
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        self.V = initial_value

    def iteration(self) -> None:
        self.assert_not_converged()

        # For each state
        delta = 0.0
        for state in range(self.n_states):

            # Find maximum action-value
            max_action_value = np.max(self.one_step_lookahead(state))

            # Update the value function
            update = max_action_value - self.V.of(state)
            delta = max(delta, abs(update))
            self.V.update(state, update)
        
        # If converged, compute the deterministic optimal policy
        if delta < self.theta:
            self.pi = self.policy
            self.converged = True

    @property
    def value(self) -> TabularStateValue:
        return self.V

    @property
    def policy(self) -> TabularPolicy:
        if not self.converged:
            self.pi = TabularPolicy(self.n_states, self.n_actions)
            for state in range(self.n_states):
                best_action = np.argmax(self.one_step_lookahead(state))
                self.pi.make_deterministic(state, best_action)
        return self.pi
