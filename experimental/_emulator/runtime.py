from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from ..parallel_dims import Group
from .greenlet_backend import Completed, GreenletBackend, Suspended, suspend


@dataclass(frozen=True)
class _Collective:
    operation: str
    group: Group
    tensor: Tensor
    dim: int = 0


@dataclass(frozen=True)
class _Send:
    destination: int
    tensor: Tensor
    tag: str


@dataclass(frozen=True)
class _Receive:
    source: int
    tag: str


@dataclass(frozen=True)
class _Backward:
    loss: Tensor | None


@dataclass
class _PendingCollective:
    signature: tuple[str, int]
    tensors: dict[int, Tensor]


class DistributedRuntime:
    def __init__(self, world_size: int) -> None:
        self.world_size = world_size

    def run[T](self, worker: Callable[[], T]) -> list[T]:
        backend = GreenletBackend(self.world_size, worker)
        ready = deque((rank, None) for rank in range(self.world_size))
        results: list[T | None] = [None] * self.world_size
        completed = 0
        sequence: defaultdict[tuple[int, Group], int] = defaultdict(int)
        collectives: dict[tuple[Group, int], _PendingCollective] = {}
        sends: defaultdict[tuple[int, int, str], deque[Tensor]] = defaultdict(deque)
        receives: defaultdict[tuple[int, int, str], deque[int]] = defaultdict(deque)
        backward_waiters: dict[int, Tensor | None] = {}

        while ready:
            rank, value = ready.popleft()
            step = backend.resume(rank, value)

            if isinstance(step, Completed):
                results[rank] = step.value
                completed += 1
                continue

            request = cast(Suspended, step).value
            if request is None:
                ready.append((rank, None))
            elif isinstance(request, _Collective):
                self._handle_collective(rank, request, sequence, collectives, ready)
            elif isinstance(request, _Send):
                key = (rank, request.destination, request.tag)
                if receives[key]:
                    receiver = receives[key].popleft()
                    ready.append((rank, None))
                    ready.append((receiver, request.tensor))
                else:
                    sends[key].append(request.tensor)
            elif isinstance(request, _Receive):
                key = (request.source, rank, request.tag)
                if sends[key]:
                    tensor = sends[key].popleft()
                    ready.append((request.source, None))
                    ready.append((rank, tensor))
                else:
                    receives[key].append(rank)
            elif isinstance(request, _Backward):
                backward_waiters[rank] = request.loss
                if len(backward_waiters) == self.world_size:
                    losses = [loss for loss in backward_waiters.values() if loss is not None]
                    if not losses:
                        raise RuntimeError("distributed backward needs at least one loss")
                    loss = torch.stack(losses).mean()
                    loss.backward()
                    ready.extend((member, loss.item()) for member in range(self.world_size))
                    backward_waiters.clear()
            else:
                raise RuntimeError(f"rank {rank} suspended with unsupported value {request!r}")

        if completed != self.world_size:
            waiting_collectives = [
                f"{group.name}{group.ranks} step {index}: {sorted(pending.tensors)}"
                for (group, index), pending in collectives.items()
            ]
            waiting_messages = [
                f"{source}->{destination} tag={tag!r}"
                for source, destination, tag in sends.keys() | receives.keys()
                if sends[(source, destination, tag)] or receives[(source, destination, tag)]
            ]
            details = "; ".join(waiting_collectives + waiting_messages)
            raise RuntimeError(f"distributed program deadlocked: {details}")

        return cast(list[T], results)

    def _handle_collective(
        self,
        rank: int,
        request: _Collective,
        sequence: defaultdict[tuple[int, Group], int],
        collectives: dict[tuple[Group, int], _PendingCollective],
        ready: deque[tuple[int, object]],
    ) -> None:
        if rank not in request.group.ranks:
            raise RuntimeError(f"rank {rank} is not in group {request.group.ranks}")

        rank_sequence = (rank, request.group)
        index = sequence[rank_sequence]
        sequence[rank_sequence] += 1
        key = (request.group, index)
        signature = request.operation, request.dim

        if key not in collectives:
            collectives[key] = _PendingCollective(signature, {})
        pending = collectives[key]
        if pending.signature != signature:
            raise RuntimeError(
                f"collective mismatch in {request.group.name}{request.group.ranks} at step {index}: "
                f"expected {pending.signature}, received {signature} from rank {rank}"
            )
        pending.tensors[rank] = request.tensor

        if len(pending.tensors) == request.group.size:
            outputs = self._evaluate_collective(request, pending.tensors)
            del collectives[key]
            ready.extend((member, outputs[member]) for member in pending.tensors)

    @staticmethod
    def _evaluate_collective(request: _Collective, tensors: dict[int, Tensor]) -> dict[int, Tensor]:
        ordered = [tensors[rank] for rank in request.group.ranks]

        if request.operation == "all_reduce":
            result = torch.stack(ordered).sum(dim=0)
            return {rank: result.clone() for rank in request.group.ranks}

        if request.operation == "all_gather":
            result = torch.cat(ordered, dim=request.dim)
            return {rank: result.clone() for rank in request.group.ranks}

        if request.operation == "reduce_scatter":
            result = torch.stack(ordered).sum(dim=0)
            if result.shape[request.dim] % request.group.size:
                raise RuntimeError("reduce_scatter dimension must be divisible by the group size")
            chunks = result.chunk(request.group.size, dim=request.dim)
            return {rank: chunk.clone() for rank, chunk in zip(request.group.ranks, chunks, strict=True)}

        if request.operation == "all_to_all":
            chunks_by_source = []
            for tensor in ordered:
                if tensor.shape[request.dim] % request.group.size:
                    raise RuntimeError("all_to_all dimension must be divisible by the group size")
                chunks_by_source.append(tensor.chunk(request.group.size, dim=request.dim))
            return {
                rank: torch.cat([chunks[destination] for chunks in chunks_by_source], dim=request.dim)
                for destination, rank in enumerate(request.group.ranks)
            }

        raise RuntimeError(f"unknown collective {request.operation!r}")


def _collective(operation: str, tensor: Tensor, group: Group, dim: int = 0) -> Tensor:
    if group.size == 1:
        return tensor
    return cast(Tensor, suspend(_Collective(operation, group, tensor, dim)))


def send(tensor: Tensor, destination: int, tag: str) -> None:
    suspend(_Send(destination, tensor, tag))


def receive(source: int, tag: str) -> Tensor:
    return cast(Tensor, suspend(_Receive(source, tag)))


def all_reduce(tensor: Tensor, group: Group) -> Tensor:
    return _collective("all_reduce", tensor, group)


def all_gather(tensor: Tensor, group: Group, dim: int = 0) -> Tensor:
    return _collective("all_gather", tensor, group, dim)


def reduce_scatter(tensor: Tensor, group: Group, dim: int = 0) -> Tensor:
    return _collective("reduce_scatter", tensor, group, dim)


def all_to_all(tensor: Tensor, group: Group, dim: int = 0) -> Tensor:
    return _collective("all_to_all", tensor, group, dim)


def distributed_backward(loss: Tensor | None) -> float:
    return cast(float, suspend(_Backward(loss)))
