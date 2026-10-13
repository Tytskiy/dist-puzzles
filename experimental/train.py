from collections import defaultdict
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from .emulator import backward, get_rank, run
from .model import ModelConfig, TransformerBlock
from .parallel_dims import ParallelDims
from .parallelisms.data import FSDP
from .parallelisms.pipeline import receive_pipeline_input, send_pipeline_output
from .parallelize import parallelize_model, sync_gradients


@dataclass(frozen=True)
class TrainResult:
    stage: int
    loss: float | None
    attention_shard: float
    expert_shard: float


def make_input(dims: ParallelDims, config: ModelConfig, local_sequence: int) -> Tensor:
    coordinate = dims.coordinate(get_rank())
    sequence = torch.arange(local_sequence, dtype=torch.float32) + coordinate["cp"] * local_sequence
    features = torch.arange(config.hidden_size, dtype=torch.float32)
    lane = 100 * coordinate["dp_replicate"] + coordinate["ep"]
    return torch.sin((sequence[:, None] + features[None, :] + lane) / 10).unsqueeze(0)


def train_step(dims: ParallelDims, config: ModelConfig) -> TrainResult:
    rank = get_rank()
    stage = dims.coordinate(rank)["pp"]
    stage_config = replace(config, seed=config.seed + 10_000 * stage)
    model = parallelize_model(TransformerBlock(stage_config), dims)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    activation = (
        make_input(dims, stage_config, local_sequence=4) if stage == 0 else receive_pipeline_input(dims)
    )
    output = model(activation)

    if stage + 1 < dims.pp:
        send_pipeline_output(output, dims)
        loss = None
    else:
        loss = output.square().mean()

    global_loss = backward(
        loss,
        replicated_over=(
            dims.group(rank, "dp_shard"),
            dims.group(rank, "tp"),
        ),
    )
    sync_gradients(model, dims)
    optimizer.step()

    if not isinstance(model.attention, FSDP) or not isinstance(model.experts, FSDP):
        raise TypeError("parallelized modules must be FSDP instances")
    return TrainResult(
        stage=stage,
        loss=global_loss if stage + 1 == dims.pp else None,
        attention_shard=model.attention.shard.detach().sum().item(),
        expert_shard=model.experts.shard.detach().sum().item(),
    )


def _self_check() -> None:
    dims = ParallelDims(dp_replicate=2, dp_shard=2, cp=2, tp=2, ep=2, pp=2)
    config = ModelConfig()
    results = run(dims, lambda: train_step(dims, config))

    assert len(results) == dims.world_size
    assert all(result.loss is None for result in results if result.stage == 0)
    assert all(result.loss is not None for result in results if result.stage == 1)

    attention_replicas: dict[tuple[int, int, int], set[float]] = defaultdict(set)
    expert_replicas: dict[tuple[int, int, int, int], set[float]] = defaultdict(set)
    for rank, result in enumerate(results):
        coordinate = dims.coordinate(rank)
        attention_key = coordinate["pp"], coordinate["dp_shard"], coordinate["tp"]
        expert_key = attention_key + (coordinate["ep"],)
        attention_replicas[attention_key].add(round(result.attention_shard, 6))
        expert_replicas[expert_key].add(round(result.expert_shard, 6))

    assert all(len(values) == 1 for values in attention_replicas.values())
    assert all(len(values) == 1 for values in expert_replicas.values())


if __name__ == "__main__":
    _self_check()
