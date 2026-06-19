import flock
import torch
from flock import Work
from flock.types import Rank


def isend(dst: Rank, tensor: torch.Tensor, tag: int = 0) -> Work[None]:
    return flock.isend(dst, tensor.clone(), tag=tag)


def irecv(src: Rank, tag: int = 0) -> Work[torch.Tensor]:
    return flock.irecv(src, tag=tag, expected_type=torch.Tensor)


async def send(dst: Rank, tensor: torch.Tensor, tag: int = 0) -> None:
    await flock.send(dst, tensor.clone(), tag=tag)


async def recv(src: Rank, tag: int = 0) -> torch.Tensor:
    return await flock.recv(src, tag=tag, expected_type=torch.Tensor)
