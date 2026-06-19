from collections.abc import Callable, Coroutine
from typing import Any

import flock
from flock.scheduler import Random


def check_puzzle(
    puzzle: Callable[[], Coroutine[Any, Any, object]],
    expected: object,
    *,
    workers: int,
    seed: int | None = 0,
) -> None:
    policy = Random(seed=seed) if seed is not None else Random(seed=None)
    decorated = flock.distribute(workers=workers, policy=policy)(puzzle)
    result = decorated()
    assert result == expected, f"expected {expected!r}, got {result!r}"
