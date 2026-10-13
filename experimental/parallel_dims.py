from dataclasses import dataclass
from math import prod
from typing import ClassVar


@dataclass(frozen=True)
class Group:
    name: str
    ranks: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.ranks)

    def local_rank(self, rank: int) -> int:
        return self.ranks.index(rank)


@dataclass(frozen=True)
class ParallelDims:
    dp_replicate: int = 1
    dp_shard: int = 1
    cp: int = 1
    tp: int = 1
    ep: int = 1
    pp: int = 1

    axes: ClassVar[tuple[str, ...]] = ("pp", "dp_replicate", "dp_shard", "cp", "tp", "ep")

    def __post_init__(self) -> None:
        for axis in self.axes:
            if getattr(self, axis) < 1:
                raise ValueError(f"{axis} must be positive")

    @property
    def world_size(self) -> int:
        return prod(getattr(self, axis) for axis in self.axes)

    def coordinate(self, rank: int) -> dict[str, int]:
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank {rank} is outside world size {self.world_size}")

        coordinate: dict[str, int] = {}
        remainder = rank
        for axis in reversed(self.axes):
            size = getattr(self, axis)
            coordinate[axis] = remainder % size
            remainder //= size
        return coordinate

    def rank(self, coordinate: dict[str, int]) -> int:
        rank = 0
        for axis in self.axes:
            value = coordinate[axis]
            size = getattr(self, axis)
            if not 0 <= value < size:
                raise ValueError(f"{axis} coordinate {value} is outside size {size}")
            rank = rank * size + value
        return rank

    def group(self, rank: int, axis: str) -> Group:
        if axis not in self.axes:
            raise ValueError(f"unknown parallel axis {axis!r}")

        coordinate = self.coordinate(rank)
        ranks = tuple(self.rank(coordinate | {axis: value}) for value in range(getattr(self, axis)))
        return Group(axis, ranks)

    def neighbor(self, rank: int, axis: str, offset: int) -> int:
        coordinate = self.coordinate(rank)
        coordinate[axis] += offset
        return self.rank(coordinate)
