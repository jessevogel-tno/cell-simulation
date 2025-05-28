from numpy.random import RandomState
from tqdm import tqdm

from risq.method import Method
from risq.model import Model, State


class MonteCarlo2D(Method):

    def __init__(
        self,
        model: Model,
        num_trials: int,
        width: int,
        height: int,
        time_steps: int,
        seed: int | None = None,
    ):
        self.model = model
        self.num_trials = num_trials
        self.width = width
        self.height = height
        self.time_steps = time_steps
        self.num_cells = self.width * self.height

        self.results_sum_cells = None
        self.results_sum_square_cells = None
        self.results_final_counts = None

        self.random_state = RandomState(seed)

    def name() -> str:
        return "Monte Carlo"

    def probability(self, time: int, state: State) -> float:
        if time == 0:
            return 1.0 if state == 0 else 0.0

        if self.results_sum_cells is None:
            self.simulate()

        return self.results_sum_cells[time - 1][state] / (
            self.num_trials * self.width * self.height
        )

    def variance(self, time: int, state: State) -> float:
        if time == 0:
            return 0.0

        if self.results_sum_cells is None:
            self.simulate()

        count_per_trial = self.results_sum_cells[time - 1][state] / self.num_trials
        square_count_per_trial = (
            self.results_sum_square_cells[time - 1][state] / self.num_trials
        )
        variance = square_count_per_trial - count_per_trial**2
        variance_per_cell = variance / (self.width * self.height)

        return variance_per_cell

    def final_counts(self, state: State) -> list[int]:
        """Returns a list (of length `self.num_trials`) of the number of cells in given state, for each trial."""
        if self.results_final_counts is None:
            self.simulate()

        return [
            self.results_final_counts[trial][state] for trial in range(self.num_trials)
        ]

    def simulate_cell(
        self, time: int, cells: list[list[State]], x: int, y: int
    ) -> State:
        old = cells[x][y]

        # Cell is overgrown by neighbors with certain probabilities
        neighbors = [
            cells[y][(x + 1) % self.width],
            cells[y][(x - 1) % self.width],
            cells[(y + 1) % self.height][x],
            cells[(y - 1) % self.height][x],
        ]
        self.random_state.shuffle(neighbors)
        for i in range(4):
            q = self.model.prob_spread(time, neighbors[i], old)
            if q > 0.0 and self.random_state.random() < q:
                return neighbors[i]

        # If not overgrown, cell changes according to internal probabilities
        return self.random_state.choice(
            self.model.states,
            p=[self.model.prob_internal(time, new, old) for new in self.model.states],
        )

    def simulate(self):
        self.results_sum_cells = [
            [0 for _ in range(self.model.num_states)] for _ in range(self.time_steps)
        ]
        self.results_sum_square_cells = [
            [0 for _ in range(self.model.num_states)] for _ in range(self.time_steps)
        ]

        self.results_final_counts = []

        for _ in tqdm(range(self.num_trials), leave=False):
            # Start trial with cells all in state 0
            cells = [[0 for x in range(self.width)] for y in range(self.height)]

            for t in range(self.time_steps):
                # Simulate one time step
                cells = [
                    [self.simulate_cell(t + 1, cells, x, y) for x in range(self.width)]
                    for y in range(self.height)
                ]

                # Update `self.results`
                for state in self.model.states:
                    count = len(
                        [
                            1
                            for y in range(self.height)
                            for x in range(self.width)
                            if cells[y][x] == state
                        ]
                    )
                    self.results_sum_cells[t][state] += count
                    self.results_sum_square_cells[t][state] += count**2

            self.results_final_counts.append(
                [
                    len(
                        [
                            1
                            for y in range(self.height)
                            for x in range(self.width)
                            if cells[y][x] == state
                        ]
                    )
                    for state in self.model.states
                ]
            )
