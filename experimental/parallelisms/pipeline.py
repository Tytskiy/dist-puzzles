from torch import Tensor

from ..emulator import get_rank, receive, send
from ..parallel_dims import ParallelDims


def receive_pipeline_input(dims: ParallelDims) -> Tensor:
    rank = get_rank()
    return receive(dims.neighbor(rank, "pp", -1), "pipeline activation")


def send_pipeline_output(tensor: Tensor, dims: ParallelDims) -> None:
    rank = get_rank()
    send(tensor, dims.neighbor(rank, "pp", 1), "pipeline activation")
