from risq.model import Model


def create_two_state_model(prob_mutate: float, prob_spread: float):
    """Creates cell model with states H and C.

    In the toy model, a cell can be in one of two states: H (healthy) and C (cancerous).
    Healthy cells can become cancerous with a probability `prob_mutate` and cancerous
    cells can overgrow healthy cells with a probability `prob_spread`
    """
    num_states = 2  # healthy and cancerous
    num_neighbors = 4  # in a 2 dimensional grid, each cell has 4 neighbors

    p = prob_mutate
    q = prob_spread

    probs_internal = [
        [1.0 - p, 0.0],  # H
        [p, 1.0],  # C
    ]

    probs_spread = [
        [0.0, 0.0],  # H
        [q, 0.0],  # C
    ]

    return Model(
        num_states=num_states,
        num_neighbors=num_neighbors,
        probs_internal=probs_internal,
        probs_spread=probs_spread,
    )


def create_six_mutations_model(
    prob_mutate: float,
    prob_dying: float,
    prob_spread: float,
):
    """Creates cell model with states H, S1, S2, S3, S4, S5, S6, C and D."""
    num_states = 9
    num_neighbors = 4  # in a 2 dimensional grid, each cell has 4 neighbors
    labels = ["H", "S1", "S2", "S3", "S4", "S5", "S6", "C", "D"]

    x = prob_mutate
    y = prob_dying
    z = prob_spread

    _i_ = 1.0 - x - y
    _c_ = 1.0

    r = 1.0  # revival (D -> H)
    _d_ = 1.0 - r

    probs_internal = [
        [_i_, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, r],  # H
        [x, _i_, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # S1
        [0.0, x, _i_, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # S2
        [0.0, 0.0, x, _i_, 0.0, 0.0, 0.0, 0.0, 0.0],  # S3
        [0.0, 0.0, 0.0, x, _i_, 0.0, 0.0, 0.0, 0.0],  # S4
        [0.0, 0.0, 0.0, 0.0, x, _i_, 0.0, 0.0, 0.0],  # S5
        [0.0, 0.0, 0.0, 0.0, 0.0, x, _i_, 0.0, 0.0],  # S6
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, x, _c_, 0.0],  # C
        [y, y, y, y, y, y, y, 0.0, _d_],  # D
    ]

    probs_spread = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # H
        [1 * z, 1 * z, 1 * z, 1 * z, 1 * z, 1 * z, 1 * z, 1 * z, 1 * z],  # S1
        [2 * z, 2 * z, 2 * z, 2 * z, 2 * z, 2 * z, 2 * z, 2 * z, 2 * z],  # S2
        [3 * z, 3 * z, 3 * z, 3 * z, 3 * z, 3 * z, 3 * z, 3 * z, 3 * z],  # S3
        [4 * z, 4 * z, 4 * z, 4 * z, 4 * z, 4 * z, 4 * z, 4 * z, 4 * z],  # S4
        [5 * z, 5 * z, 5 * z, 5 * z, 5 * z, 5 * z, 5 * z, 5 * z, 5 * z],  # S5
        [6 * z, 6 * z, 6 * z, 6 * z, 6 * z, 6 * z, 6 * z, 6 * z, 6 * z],  # S6
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # C
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # D
    ]

    return Model(
        num_states=num_states,
        num_neighbors=num_neighbors,
        probs_internal=probs_internal,
        probs_spread=probs_spread,
        labels=labels,
    )
