from risq.method import Method
from risq.model import Model, State


class SingleCellMethod(Method):

    def __init__(self, model: Model):
        self.model = model
        self.memory = {}

    def name() -> str:
        return "Single cell"

    def probability(self, time: int, state: State) -> float:
        for t in range(0, time, 16):  # (to prevent max recursion exceeded)
            self.compute_probability(t, state)
        return self.compute_probability(time, state)

    def variance(self, time: int, state: State) -> float:
        p = self.probability(time, state)
        return p - p**2

    def compute_probability(self, time: int, state: State) -> float:
        # Recall result from memory if possible
        key = (time, state)
        if key in self.memory:
            return self.memory[key]

        # Otherwise, compute the probability and store in memory
        prob = self._compute_probability(time, state)
        self.memory[key] = prob
        return prob

    def _compute_probability(self, time: int, state: State) -> float:
        X = state

        # Base case: at time 0 all cells are healthy
        if time == 0:
            return self.model.prob_initial((X,))

        # For stability reasons: compute prob of last state as 1 - sum prob other states
        if X == self.model.states[-1]:
            return 1.0 - sum(
                self.compute_probability(time, U) for U in self.model.states[:-1]
            )

        p = 0.0
        for Y in self.model.states:
            # Probability that cell was in state Y previous time step
            p_Y = self.compute_probability(time - 1, Y)

            if p_Y == 0.0:
                continue

            # Probability that Y is overgrown by some neighboring X
            p_X = self.compute_probability(time - 1, X)
            p_X_overgrows_Y = self.model.prob_spread(time, X, Y)
            p_Y_overgrown_by_some_X = (
                1.0 - (1.0 - p_X * p_X_overgrows_Y) ** self.model.num_neighbors
            )
            p += p_Y * p_Y_overgrown_by_some_X

            # When Y is not overgrown by any neighbor, look at the internal probabilities
            p_X_from_Y = self.model.prob_internal(time, X, Y)
            p_Y_not_overgrown = 0.0  # i.e. by a single neighbor
            for Z in self.model.states:
                p_Z = self.compute_probability(time - 1, Z)
                p_Z_overgrows_Y = self.model.prob_spread(time, Z, Y)
                p_Y_not_overgrown += p_Z * (1.0 - p_Z_overgrows_Y)
            p_Y_not_overgrown_at_all = p_Y_not_overgrown**self.model.num_neighbors
            p += p_Y * p_Y_not_overgrown_at_all * p_X_from_Y

        return p
