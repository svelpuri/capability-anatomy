from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Callable, Generic, Iterable, TypeVar


ValueT = TypeVar("ValueT")


@dataclass(frozen=True)
class Measurement(Generic[ValueT]):
    value: ValueT
    elapsed_seconds: float
    peak_memory_bytes: int | None
    input_tokens: int | None
    output_tokens: int | None


class MeasurementControls:
    def __init__(
        self,
        *,
        seed: int,
        warmup_runs: int,
        repetitions: int,
        randomized_order: bool,
        synchronize: Callable[[], None] = lambda: None,
        memory_bytes: Callable[[], int | None] = lambda: None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if warmup_runs < 0 or repetitions <= 0:
            raise ValueError("measurement controls require non-negative warmups and positive repetitions")
        self.seed = seed
        self.warmup_runs = warmup_runs
        self.repetitions = repetitions
        self.randomized_order = randomized_order
        self.synchronize = synchronize
        self.memory_bytes = memory_bytes
        self.clock = clock

    def order(self, values: Iterable[ValueT]) -> tuple[ValueT, ...]:
        ordered = list(values)
        if self.randomized_order:
            random.Random(self.seed).shuffle(ordered)
        return tuple(ordered)

    def measure(
        self,
        operation: Callable[[], ValueT],
        token_counts: Callable[[ValueT], tuple[int | None, int | None]] = lambda _value: (None, None),
    ) -> tuple[Measurement[ValueT], ...]:
        for _ in range(self.warmup_runs):
            self.synchronize()
            operation()
            self.synchronize()
        measurements = []
        for _ in range(self.repetitions):
            self.synchronize()
            started = self.clock()
            value = operation()
            self.synchronize()
            elapsed = self.clock() - started
            if elapsed < 0:
                raise RuntimeError("measurement clock moved backwards")
            input_tokens, output_tokens = token_counts(value)
            measurements.append(
                Measurement(value, elapsed, self.memory_bytes(), input_tokens, output_tokens)
            )
        return tuple(measurements)
