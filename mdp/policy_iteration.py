__all__ = [
    'PolicyIteration',
]


from copy import deepcopy

from gym import Env
import numpy as np

from mdp import (
    DPMethod,
    TabularStateValue, TabularPolicy,
    one_step_lookahead,
)


class PolicyIteration(DPMethod):
    """
    Policy evaluation: Bellman expectation backup.
    Policy improvement: Greedy policy improvement.
    """

    def __init__(
        self,
        starting_policy: TabularPolicy,
        n_states: int,
        n_actions: int,
        P: dict,
        **kwargs,
    ) -> None:
        super().__init__(n_states, n_actions, P, **kwargs)
        self.pi = deepcopy(starting_policy)

    def iteration(self) -> bool:
        assert self.converged is False, 'Convergence already reached'

        # Evaluate current policy
        self.iterative_policy_evaluation()

        # Improve policy
        policy_stable = self.policy_improvement()

        # If no changes to the policy have been made, the current policy is optimal
        if policy_stable:
            self.converged = True
        return policy_stable

    def iterative_policy_evaluation(self) -> TabularStateValue:
        """
        Return value approximated to true value of the current policy.
        """
        self.V = TabularStateValue(self.n_states)
        while True:

            delta = 0.
            for state in range(self.n_states):

                # Compute the value of this state
                state_value = 0.
                for action, action_prob in enumerate(self.pi.probabilities(state)):
                    for trans_prob, next_state, reward, _ in self.P[state][action]:
                        state_value += action_prob * trans_prob * (
                            reward + self.gamma * self.V.of(next_state)
                        )

                # Update the value function
                update = state_value - self.V.of(state)
                delta = max(delta, abs(update))
                self.V.update(state, update)

            # Stop evaluation when the approximation is accurate enough
            if delta < self.theta:
                return

    def policy_improvement(self) -> bool:
        """
        Improve policy towards optimality.
        """
        policy_stable = True
        for state in range(self.n_states):

            # Find the best action under the current policy
            old_action = self.pi.sample_greedy(state)

            # Find the best action according to the current value
            action_values = one_step_lookahead(self.n_actions, self.P, state, self.V, self.gamma)
            best_action = np.argmax(action_values)

            # Update the policy
            if old_action != best_action:
                policy_stable = False
            self.pi.make_deterministic(state, best_action)

        return policy_stable

    @property
    def value(self) -> TabularStateValue:
        if not hasattr(self, 'V'):
            self.iterative_policy_evaluation()
        return self.V

    @property
    def policy(self) -> TabularPolicy:
        return self.pi
