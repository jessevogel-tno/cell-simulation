State = int  # type alias State to int


class Model:

    def __init__(
        self,
        num_states: int,
        num_neighbors: int,
        probs_internal: list[list[float]],
        probs_spread: list[list[float]],
        labels: list[str] | None = None,
        colors: list[str] | None = None,
    ) -> None:
        """
        Args:
            num_states: The number of states a cell can be in.
            num_neighbors: The number of neighbors a single cell has.
            probs_internal: The internal transition probabilities of a cell.
                E.g. `probs_internal[i][j]` is the probability a cell in state `j`
                transitions to state `i` in one time step. This should be a Markov matrix.
            probs_spread: The spreading probabilities between cells.
                E.g. `probs_spread[i][j]` is the probability a cell in state `j` is
                overgrown by a neighboring cell in state `i` in one time step.
        """
        self.num_states = num_states
        self.num_neighbors = num_neighbors
        self.probs_internal = probs_internal
        self.probs_spread = probs_spread
        self.labels = labels or [str(i) for i in range(num_states)]
        self.colors = colors

        self.states: list[State] = list(range(num_states))

        self.validate()

    def prob_initial(self, pattern: tuple[State, ...]):
        """Initially, all cells are assumed to be in state 0."""
        return 1.0 if all(x == 0 for x in pattern) else 0.0

    def prob_internal(self, time: float, new: State, old: State) -> float:
        """Internal probability that a cell goes from state `old` to state `new`."""
        return self.probs_internal[new][old]

    def prob_spread(self, time: float, attacker: State, target: State) -> float:
        """Probability that a cell in state `attacker` overgrows a neighboring cell in state `target`."""
        return self.probs_spread[attacker][target]

    def validate(self) -> float:
        # Check dimensions of matrices
        assert len(self.probs_internal) == self.num_states and all(
            len(row) == self.num_states for row in self.probs_internal
        ), f"`probs_internal` is expected to be a {self.num_states} x {self.num_states} matrix"

        assert len(self.probs_spread) == self.num_states and all(
            len(row) == self.num_states for row in self.probs_spread
        ), f"`probs_spread` is expected to be a {self.num_states} x {self.num_states} matrix"

        # Check Markov Chain condition: all columns of `probs_internal` should add up to 1
        for i in range(self.num_states):
            sum_column_i = sum(
                self.probs_internal[j][i] for j in range(self.num_states)
            )
            assert (
                abs(sum_column_i - 1.0) < 1e-8
            ), f"Column {i + 1} of `probs_internal` should add up to 1.0 (adds up to {sum_column_i})"

        # Check labels
        assert (
            len(self.labels) == self.num_states
        ), f"Number of labels does not match number of states ({len(self.labels)} != {self.num_states})"

        # Check colors
        assert (
            self.colors is None or len(self.colors) == self.num_states
        ), f"Number of colors does not match number of states ({len(self.colors)} != {self.num_states})"
