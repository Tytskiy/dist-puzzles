from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass
from typing import cast

import torch
from greenlet import getcurrent, greenlet
from torch.utils._python_dispatch import TorchDispatchMode

from .torch_tls import capture as capture_torch_tls

_rank: ContextVar[int] = ContextVar("rank")
_world_size: ContextVar[int] = ContextVar("world_size")


@dataclass(frozen=True)
class Suspended:
    rank: int
    value: object


@dataclass(frozen=True)
class Completed[T]:
    rank: int
    value: T


type Step[T] = Suspended | Completed[T]


@dataclass(frozen=True)
class _TorchThreadState:
    state: object

    @classmethod
    def capture(cls) -> "_TorchThreadState":
        return cls(capture_torch_tls())

    def restore(self) -> None:
        self.state.restore()

    @contextmanager
    def active(self) -> Iterator[None]:
        previous = self.capture()
        self.restore()
        try:
            yield
        finally:
            previous.restore()


@dataclass
class _Worker:
    task: greenlet
    torch_state: _TorchThreadState
    started: bool = False


class GreenletBackend[T]:
    def __init__(self, world_size: int, worker: Callable[[], T]) -> None:
        if world_size < 1:
            raise ValueError("world_size must be positive")

        self._parent = getcurrent()
        initial_torch_state = _TorchThreadState.capture()
        self._workers: list[_Worker] = []

        for rank in range(world_size):
            task = greenlet(self._entry(worker, rank, world_size), parent=self._parent)
            task.gr_context = copy_context()
            self._workers.append(_Worker(task, initial_torch_state))

    @staticmethod
    def _entry(worker: Callable[[], T], rank: int, world_size: int) -> Callable[[], T]:
        def run() -> T:
            rank_token = _rank.set(rank)
            world_size_token = _world_size.set(world_size)
            try:
                return worker()
            finally:
                _world_size.reset(world_size_token)
                _rank.reset(rank_token)

        return run

    def resume(self, rank: int, value: object = None) -> Step[T]:
        if getcurrent() is not self._parent:
            raise RuntimeError("workers must be resumed by the greenlet that created the backend")

        try:
            worker = self._workers[rank]
        except IndexError:
            raise ValueError(f"rank {rank} is outside world size {len(self._workers)}") from None

        if rank < 0:
            raise ValueError(f"rank {rank} is outside world size {len(self._workers)}")
        if worker.task.dead:
            raise RuntimeError(f"rank {rank} has already completed")

        with worker.torch_state.active():
            output = worker.task.switch(value) if worker.started else worker.task.switch()
            worker.started = True
            worker.torch_state = _TorchThreadState.capture()

        if worker.task.dead:
            return Completed(rank, cast(T, output))
        return Suspended(rank, output)

    def is_completed(self, rank: int) -> bool:
        return self._workers[rank].task.dead


def get_rank() -> int:
    try:
        return _rank.get()
    except LookupError:
        raise RuntimeError("get_rank() must be called from a worker") from None


def get_world_size() -> int:
    try:
        return _world_size.get()
    except LookupError:
        raise RuntimeError("get_world_size() must be called from a worker") from None


def suspend(value: object = None) -> object:
    try:
        _rank.get()
    except LookupError:
        raise RuntimeError("suspend() must be called from a worker") from None
    return getcurrent().parent.switch(value)


def _self_check() -> None:
    dispatches: list[int] = []

    class RecordingMode(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            function: Callable[..., object],
            types: object,
            args: tuple[object, ...],
            kwargs: dict[str, object] | None,
        ) -> object:
            dispatches.append(get_rank())
            return function(*args, **(kwargs or {}))

    def worker() -> tuple[int, bool, bool]:
        if get_rank() == 0:
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16), RecordingMode():
                assert suspend("rank 0 paused") == "resume rank 0"
                torch.ones(1) + 1
                return get_rank(), torch.is_grad_enabled(), torch.is_autocast_enabled("cpu")

        with torch.inference_mode():
            assert suspend("rank 1 paused") == "resume rank 1"
            torch.ones(1) + 1
            return get_rank(), torch.is_inference_mode_enabled(), torch.is_grad_enabled()

    backend = GreenletBackend(2, worker)

    assert backend.resume(0) == Suspended(0, "rank 0 paused")
    assert torch.is_grad_enabled()
    assert not torch.is_autocast_enabled("cpu")
    assert backend.resume(1) == Suspended(1, "rank 1 paused")
    assert not torch.is_inference_mode_enabled()
    assert backend.resume(0, "resume rank 0") == Completed(0, (0, False, True))
    assert backend.resume(1, "resume rank 1") == Completed(1, (1, True, False))
    assert dispatches and set(dispatches) == {0}


if __name__ == "__main__":
    _self_check()
