import flock
import pytest
import torch

import dist_puzzles as dp


def test_send_recv_tensor() -> None:
    @flock.distribute(workers=2)
    async def ping() -> torch.Tensor | None:
        if flock.get_rank() == 0:
            tensor = torch.tensor([1.0, 2.0])
            await dp.send(1, tensor)
            tensor.add_(10.0)
            return None
        return await dp.recv(0)

    result = ping()
    assert result[0] is None
    assert torch.equal(result[1], torch.tensor([1.0, 2.0]))


def test_isend_irecv_tensor() -> None:
    @flock.distribute(workers=2)
    async def ping() -> torch.Tensor | None:
        if flock.get_rank() == 0:
            work = dp.isend(1, torch.tensor([3.0]))
            await work.wait()
            return None

        work = dp.irecv(0)
        return await work.wait()

    result = ping()
    assert result[0] is None
    assert torch.equal(result[1], torch.tensor([3.0]))


def test_recv_respects_tag() -> None:
    @flock.distribute(workers=2)
    async def ping() -> torch.Tensor | None:
        if flock.get_rank() == 0:
            first = dp.isend(1, torch.tensor([1.0]), tag=0)
            second = dp.isend(1, torch.tensor([2.0]), tag=1)
            await first.wait()
            await second.wait()
            return None
        return await dp.recv(0, tag=1)

    result = ping()
    assert result[0] is None
    assert torch.equal(result[1], torch.tensor([2.0]))


def test_recv_rejects_non_tensor() -> None:
    @flock.distribute(workers=2)
    async def ping() -> object:
        if flock.get_rank() == 0:
            await flock.send(1, "not a tensor")
            return None
        return await dp.recv(0)

    with pytest.raises(TypeError, match="expected Tensor, got str"):
        ping()


def test_irecv_rejects_non_tensor() -> None:
    @flock.distribute(workers=2)
    async def ping() -> object:
        if flock.get_rank() == 0:
            await flock.send(1, "not a tensor")
            return None
        return await dp.irecv(0).wait()

    with pytest.raises(TypeError, match="expected Tensor, got str"):
        ping()
