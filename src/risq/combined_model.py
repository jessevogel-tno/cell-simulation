from collections.abc import Callable

from risq.model import Model, State


class CombinedModel(Model):

    def __init__(
        self,
        model_default: Model,
        model_special: Model,
        special_condition: Callable[[int], bool],
    ) -> None:
        """
        Args:
            model_default: The default model, used when `special_condition` is not satisfied.
            model_special: The special model, used when `special_condition` is satisfied.
            special_condition: The condition when the special model is used.
        """
        self.model_default = model_default
        self.model_special = model_special
        self.special_condition = special_condition

    def _get_model(self, time: int) -> Model:
        return (
            self.model_special if self.special_condition(time) else self.model_default
        )

    def prob_initial(self, pattern: tuple[State, ...]):
        """Initial probabilities."""
        return self._get_model(0).prob_initial(pattern)

    def prob_internal(self, time: float, new: State, old: State) -> float:
        """Internal probability that a cell goes from state `old` to state `new`."""
        return self._get_model(time).prob_internal(time, new, old)

    def prob_spread(self, time: float, attacker: State, target: State) -> float:
        """Probability that a cell in state `attacker` overgrows a neighboring cell in state `target`."""
        return self._get_model(time).prob_spread(time, attacker, target)

    @property
    def states(self) -> list[State]:
        return self.model_default.states

    @property
    def num_neighbors(self) -> int:
        return self.model_default.num_neighbors

    @property
    def num_states(self) -> int:
        return self.model_default.num_states
