from typing import assert_type

import flock
import torch

import dist_puzzles as dp


async def _p2p() -> None:
    tensor = torch.tensor([1.0])

    assert_type(dp.isend(0, tensor), flock.Work[None])
    assert_type(await dp.isend(0, tensor).wait(), None)

    assert_type(await dp.irecv(0).wait(), torch.Tensor)

    assert_type(await dp.send(0, tensor), None)
    assert_type(await dp.recv(0), torch.Tensor)
