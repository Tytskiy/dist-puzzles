import flock
import torch

from dist_puzzles.grading import check_puzzle


def test_imports() -> None:
    assert flock.get_rank is not None
    assert torch.tensor([1.0]) is not None


def test_check_puzzle_passes() -> None:
    async def ping() -> str | None:
        if flock.get_rank() == 0:
            await flock.send(1, "ping")
            return None
        return await flock.recv(0)

    check_puzzle(ping, [None, "ping"], workers=2)
