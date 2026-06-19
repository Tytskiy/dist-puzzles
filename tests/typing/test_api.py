from typing import assert_type

import flock
import torch


async def _flock_torch() -> None:
    t = torch.tensor([1.0])
    assert_type(await flock.all_reduce(t, "sum").wait(), torch.Tensor)
