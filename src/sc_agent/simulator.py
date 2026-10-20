"""Bounded stochastic-computing simulation tool for Phase 3."""

import random

from .models import SimulationResult, SimulationSpec


class StochasticSimulator:
    """Run a reproducible unipolar multiplier without arbitrary code execution."""

    def simulate(self, spec: SimulationSpec) -> SimulationResult:
        if spec.operation != "unipolar_multiplier":
            raise ValueError(f"unsupported simulation operation: {spec.operation}")
        generator = random.Random(spec.seed)
        stream_x = [generator.random() < spec.x for _ in range(spec.length)]
        stream_y = [generator.random() < spec.y for _ in range(spec.length)]
        stream_out = [left and right for left, right in zip(stream_x, stream_y)]
        ones_x = sum(stream_x)
        ones_y = sum(stream_y)
        ones_output = sum(stream_out)
        estimate = ones_output / spec.length
        expected = spec.x * spec.y
        absolute_error = abs(estimate - expected)
        relative_error = absolute_error / abs(expected) if expected else None
        return SimulationResult(
            operation=spec.operation,
            length=spec.length,
            seed=spec.seed,
            x=spec.x,
            y=spec.y,
            expected=expected,
            estimate=estimate,
            absolute_error=absolute_error,
            relative_error=relative_error,
            ones_x=ones_x,
            ones_y=ones_y,
            ones_output=ones_output,
        )
