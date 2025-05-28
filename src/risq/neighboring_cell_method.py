import itertools

from risq.method import Method
from risq.model import Model, State


class NeighboringCellMethod(Method):

    def __init__(self, model: Model):
        self.model = model
        self.memory_prob_states = {}
        self.memory_prob_overgrown = {}
        self.variance_order = 4  # this seems sufficient

    def name() -> str:
        return "Neighboring cells"

    def probability(self, time: int, state: State):
        for t in range(0, time, 16):  # (to prevent max recursion exceeded)
            self.compute_probability(t, (state,))
        return self.compute_probability(time, (state,))

    def variance(self, time: int, state: State) -> float:
        # Probability of any cell being in given state at given time
        p = self.probability(time, state)

        # Compute variance
        var = 0.0

        # First order term
        var += p - p**2

        # Higher order terms
        for k in range(0, self.variance_order):
            # Compute probability q of pattern (state, *, ..., *, state)
            # where pattern consists of k `*` in the middle
            q = 0.0
            for middle in itertools.product(self.model.states, repeat=k):
                q += self.compute_probability(time, (state, *middle, state))

            # NOTE: This assumes a 2-dimensional topology:
            # The number of neighbors (k + 1) steps away is (1 + k) * `num_neighbors`
            var += (1 + k) * self.model.num_neighbors * (q - p**2)

        return var

    def compute_probability(self, time: int, pattern: tuple[State, ...]) -> float:
        # Sorting pattern gives a factor ~2 speedup
        if len(pattern) == 2 and pattern[1] < pattern[0]:
            pattern = (pattern[1], pattern[0])

        # Recall result from memory if possible
        key = (time, pattern)
        if key in self.memory_prob_states:
            return self.memory_prob_states[key]

        # Otherwise, compute the probability and store in memory
        prob = self._compute_probability(time, pattern)
        self.memory_prob_states[key] = prob
        return prob

    def _compute_probability(self, time: int, pattern: tuple[State, ...]) -> float:
        # Base case: at time 0 all cells are healthy
        if time == 0:
            return self.model.prob_initial(pattern)

        # Single cell
        if len(pattern) == 1:
            (X,) = pattern

            # For stability reasons: compute prob of last state as 1 - sum prob other states
            if X == self.model.states[-1]:
                return 1.0 - sum(
                    self.compute_probability(time, (U,)) for U in self.model.states[:-1]
                )

            p = 0.0
            for Y in self.model.states:
                # Probability that cell was in state Y previous time step
                p_Y = self.compute_probability(time - 1, (Y,))

                if p_Y == 0.0:
                    continue

                # Probability that Y is overgrown by some neighboring X
                p_XY = self.compute_probability(time - 1, (X, Y))
                p_X_overgrows_Y = self.model.prob_spread(time, X, Y)
                p_Y_overgrown_by_some_X = (
                    1.0
                    - (1.0 - p_XY / p_Y * p_X_overgrows_Y) ** self.model.num_neighbors
                )
                p += p_Y * p_Y_overgrown_by_some_X

                # When Y is not overgrown by any neighbor, look at the internal probabilities
                p_X_from_Y = self.model.prob_internal(time, X, Y)
                p_Y_not_overgrown = 1.0 - self.compute_probability_overgrown(time, Y)
                p_Y_not_overgrown_at_all = p_Y_not_overgrown**self.model.num_neighbors
                p += p_Y * p_Y_not_overgrown_at_all * p_X_from_Y

            return p

        # Two cells
        if len(pattern) == 2:
            X, Y = pattern

            # For stability, compute prob of last state as 1 - sum prob other states
            if Y == self.model.states[-1]:
                return self.compute_probability(time, (X,)) - sum(
                    self.compute_probability(time, (X, U))
                    for U in self.model.states[:-1]
                )
            if X == self.model.states[-1]:
                return self.compute_probability(time, (Y,)) - sum(
                    self.compute_probability(time, (U, Y))
                    for U in self.model.states[:-1]
                )

            p = 0.0
            for Z in self.model.states:
                for W in self.model.states:
                    # Probability that cells were in state (Z, W) previous time step
                    p_ZW = self.compute_probability(time - 1, (Z, W))
                    p_Z = self.compute_probability(time - 1, (Z,))
                    p_W = self.compute_probability(time - 1, (W,))

                    if p_Z == 0.0 or p_W == 0.0:
                        continue

                    # Probability that Z is overgrown by some neighboring X
                    p_XZ = self.compute_probability(time - 1, (X, Z))
                    p_X_overgrows_Z = self.model.prob_spread(time, X, Z)
                    p_Z_overgrown_by_some_X = 1.0 - (
                        (1.0 - p_XZ / p_Z * p_X_overgrows_Z)
                        ** (self.model.num_neighbors - 1)
                        * ((1.0 - p_X_overgrows_Z) if W == X else 1.0)
                    )

                    # Probability that Z is not overgrown at all
                    p_Z_not_overgrown = 1.0 - self.compute_probability_overgrown(
                        time, Z
                    )  # i.e. by a single (random) neighbor
                    p_W_overgrows_Z = self.model.prob_spread(time, W, Z)
                    p_Z_not_overgrown_at_all = p_Z_not_overgrown ** (
                        self.model.num_neighbors - 1
                    ) * (1.0 - p_W_overgrows_Z)

                    # Probability that W is overgrown by some neighboring Y
                    p_YW = self.compute_probability(time - 1, (Y, W))
                    p_Y_overgrows_W = self.model.prob_spread(time, Y, W)
                    p_W_overgrown_by_some_Y = 1.0 - (
                        (1.0 - p_YW / p_W * p_Y_overgrows_W)
                        ** (self.model.num_neighbors - 1)
                        * ((1.0 - p_Y_overgrows_W) if Z == Y else 1.0)
                    )

                    # Probability that W is not overgrown at all
                    p_W_not_overgrown = 1.0 - self.compute_probability_overgrown(
                        time, W
                    )  # i.e. by a single (random) neighbor
                    p_Z_overgrows_W = self.model.prob_spread(time, Z, W)
                    p_W_not_overgrown_at_all = p_W_not_overgrown ** (
                        self.model.num_neighbors - 1
                    ) * (1.0 - p_Z_overgrows_W)

                    # Internal probabilities
                    p_X_from_Z = self.model.prob_internal(time, X, Z)
                    p_Y_from_W = self.model.prob_internal(time, Y, W)

                    # Use:
                    # - p_Z_overgrown_by_some_X
                    # - p_W_overgrown_by_some_Y
                    # - p_Z_not_overgrown_at_all
                    # - p_W_not_overgrown_at_all
                    # - p_X_from_Z
                    # - p_Y_from_W
                    p += (
                        p_ZW
                        * (
                            p_Z_overgrown_by_some_X
                            + p_Z_not_overgrown_at_all * p_X_from_Z
                        )
                        * (
                            p_W_overgrown_by_some_Y
                            + p_W_not_overgrown_at_all * p_Y_from_W
                        )
                    )

            return p

        # More than 2 cells: split pattern into left and right
        if len(pattern) > 2:
            x, y = 1.0, 1.0
            for U, V in itertools.pairwise(pattern):
                x *= self.compute_probability(time, (U, V))
            for U in pattern[1:-1]:
                y *= self.compute_probability(time, (U,))

            return (x / y) if y > 0.0 else 0.0

        raise NotImplementedError(
            f"Could not compute probability of pattern '{pattern}' at time {time}"
        )

    def compute_probability_overgrown(self, time: int, state: State) -> float:
        """Computes the probability that a cell in given state at given time is overgrown
        by a random neighboring cell. (Going from `time - 1` to `time`).
        This is of course given that the cell is in that state at that time."""
        # Recall result from memory if possible
        key = (time, state)
        if key in self.memory_prob_overgrown:
            return self.memory_prob_overgrown[key]

        # Otherwise, compute the probability and store in memory
        prob = self._compute_probability_overgrown(time, state)
        self.memory_prob_overgrown[key] = prob
        return prob

    def _compute_probability_overgrown(self, time: int, state: State) -> float:
        Y = state
        p_Y = self.compute_probability(time - 1, (Y,))
        p_Y_not_overgrown = 0.0
        for Z in self.model.states:
            p_ZY = self.compute_probability(time - 1, (Z, Y))
            p_Z_overgrows_Y = self.model.prob_spread(time, Z, Y)
            p_Y_not_overgrown += p_ZY / p_Y * (1.0 - p_Z_overgrows_Y)
        return 1.0 - p_Y_not_overgrown
