import math

import numpy as np
from matplotlib import pyplot as plt

from risq.method import Method
from risq.model import Model, State
from risq.monte_carlo_method import MonteCarlo2D


def plot_distributions(simulations: list[Method], time: int, state: State):
    fig, ax = plt.subplots()

    num_cells = None
    for simulation in simulations:
        if isinstance(simulation, MonteCarlo2D):
            num_cells = simulation.num_cells
            final_counts = simulation.final_counts(state)
            bins = 2 * int(np.ceil(np.log2(len(final_counts)) + 1))
            ax_twin = ax.twinx()
            ax_twin.hist(final_counts, bins=bins, label=simulation.__class__.name())
            ax_twin.set_zorder(1)
            ax_twin.set_ylabel("Frequency (Monte Carlo)")

            ax.set_zorder(2)
            ax.patch.set_visible(False)  # hide the patch of ax1 to see ax_twin clearly

    if num_cells is None:
        raise "Required MonteCarlo2D for comparison"

    # Window to draw normal distributions in (histogram +/- 10%)
    x_min = min(final_counts)
    x_max = max(final_counts)
    x_width = x_max - x_min
    x_min -= x_width * 0.1
    x_max += x_width * 0.1
    x_min = max(x_min, 0.0)
    x_max = min(x_max, num_cells)

    i = 1
    for simulation in simulations:
        if isinstance(simulation, MonteCarlo2D):
            continue

        mean = simulation.probability(time, state) * num_cells
        variance = simulation.variance(time, state) * num_cells
        sigma = np.sqrt(variance)

        x = np.linspace(x_min, x_max, 1000)
        y = np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)

        ax.plot(x, y, label=simulation.__class__.name(), color=f"C{i}")
        i += 1

    ax.set_ylim(0.0)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Number of cells in state $C$")
    ax.set_ylabel("Probability density (Approximations)")
    fig.legend()


def compute_cancer_probability(
    method: Method, *, num_cells: int, time: int, state_cancer: State
) -> float:
    # Compute mean and variance per cell
    mean_per_cell = method.probability(time, state_cancer)
    variance_per_cell = method.variance(time, state_cancer)

    # Compute mean and variance for many cells
    mean = mean_per_cell * num_cells
    variance = variance_per_cell * num_cells
    sigma = np.sqrt(variance)

    # Compute probability of exceeding the threshold of 1 (assuming a normal distribution)
    threshold = 1
    prob = 0.5 * (1 - math.erf((threshold - mean) / (sigma * math.sqrt(2))))
    return prob


def print_latex_table(
    model: Model,
    method: type[Method],
    num_cells: int,
    state_cancer: State,
    distribution: list[tuple[int, float]],
):
    simulation = method(model)

    print("\\begin{tabular}{c|c|c}")
    print("    Age & Data & Prediction \\\\ \\hline")

    prev_prob_cdf = 0.0
    for age, prob in distribution:
        prob_cdf = compute_cancer_probability(
            method=simulation,
            num_cells=num_cells,
            time=age * 12,  # 1 time step = 1 month
            state_cancer=state_cancer,  # C
        )

        prob_est = prob_cdf - prev_prob_cdf
        prev_prob_cdf = prob_cdf

        value_data = int(round(prob * 100_000))
        value_prediction = int(round(prob_est * 100_000))

        print(f"    ${age}$ & ${value_data}$ & ${value_prediction}$ \\\\")

    print("\\end{tabular}")
