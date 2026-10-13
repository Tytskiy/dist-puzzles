from collections.abc import Callable, Iterable

from torch import Tensor

from ._emulator.greenlet_backend import get_rank
from ._emulator.runtime import (
    DistributedRuntime,
    all_gather,
    all_reduce,
    all_to_all,
    distributed_backward,
    receive,
    reduce_scatter,
    send,
)
from .parallel_dims import Group, ParallelDims

__all__ = (
    "all_gather",
    "all_reduce",
    "all_to_all",
    "backward",
    "get_rank",
    "receive",
    "reduce_scatter",
    "run",
    "send",
)


def run[T](dims: ParallelDims, worker: Callable[[], T]) -> list[T]:
    return DistributedRuntime(dims.world_size).run(worker)


def backward(loss: Tensor | None, *, replicated_over: Iterable[Group] = ()) -> float:
    rank = get_rank()
    if loss is not None and any(group.local_rank(rank) != 0 for group in replicated_over):
        loss = None
    return distributed_backward(loss)
