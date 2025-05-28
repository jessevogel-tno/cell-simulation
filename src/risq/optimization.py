from collections.abc import Callable

from tqdm import tqdm


def gradient_descent(
    loss_function: Callable,
    values: dict[str, float],
    *,
    learning_rate: float = 0.001,
    num_iter: int = 100,
    dx: float = 0.001,
):
    # Start with the given values
    values = dict(values)
    try:
        t = tqdm(range(num_iter))
        for _ in t:
            # Keep track of gradients with respect to all values `x`
            gradients = {x: 0.0 for x in values}

            # Compute initial value
            loss = loss_function(**values)
            t.set_postfix({"loss": loss, **values})

            # Compute gradients
            for x in values:
                eps = abs(values[x] * dx)
                new_values = dict(values)
                new_values[x] += eps
                new_loss = loss_function(**new_values)
                gradients[x] += (new_loss - loss) / eps

            # Apply gradients to values
            for x in values:
                values[x] -= learning_rate * gradients[x]

        # Compute final loss
        loss = loss_function(**values)
        print(f"┌──────────────────────┐")
        print(f"│  TRAINING COMPLETE ! │")
        print(f"│ FINAL LOSS: {loss:.6f} │")
        print(f"└──────────────────────┘")
        print(values)

        return values

    except KeyboardInterrupt:
        print("┌───────────────────────┐")
        print("│ ! KeyboardInterrupt ! │")
        print("│ PRINT CURRENT VALUES  │")
        print("└───────────────────────┘")
        print(values)

        return values


def steepest_descent(
    loss_function: Callable,
    values: dict[str, float],
    *,
    num_iter: int = 100,
    delta: float = 0.1,
):
    # Start with the given values
    values = dict(values)
    try:
        # Current loss
        current_loss = loss_function(**values)

        t = tqdm(range(num_iter))
        for _ in t:
            t.set_postfix({"loss": current_loss, "delta": delta, **values})

            some_update = False
            for x in values:
                for new_value_x in [values[x] - delta, values[x] + delta]:
                    new_values = dict(values)
                    new_values[x] = new_value_x
                    new_current_loss = loss_function(**new_values)
                    if new_current_loss < current_loss:
                        values[x] = new_value_x
                        current_loss = new_current_loss
                        some_update = True
                        break
            if not some_update:
                delta *= 0.5
            else:
                delta *= 1.1

        # Compute final loss
        loss = loss_function(**values)
        print(f"┌──────────────────────┐")
        print(f"│  TRAINING COMPLETE ! │")
        print(f"│ FINAL LOSS: {loss:.6f} │")
        print(f"└──────────────────────┘")
        print(values)

        return values

    except KeyboardInterrupt:
        print("┌───────────────────────┐")
        print("│ ! KeyboardInterrupt ! │")
        print("│ PRINT CURRENT VALUES  │")
        print("└───────────────────────┘")
        print(values)

        return values
