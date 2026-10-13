from .emulator import get_rank
from .model import Attention, MoE, TransformerBlock
from .parallel_dims import ParallelDims
from .parallelisms.context import parallelize_attention
from .parallelisms.data import FSDP, fully_shard, sync_replicated_gradients
from .parallelisms.expert import parallelize_experts


def parallelize_model(model: TransformerBlock, dims: ParallelDims) -> TransformerBlock:
    rank = get_rank()
    tensor_group = dims.group(rank, "tp")
    context_group = dims.group(rank, "cp")
    expert_group = dims.group(rank, "ep")
    shard_group = dims.group(rank, "dp_shard")

    if not isinstance(model.attention, Attention) or not isinstance(model.experts, MoE):
        raise TypeError("parallelize_model expects an ordinary TransformerBlock")

    attention = parallelize_attention(
        model.attention,
        tensor_group=tensor_group,
        context_group=context_group,
    )
    experts = parallelize_experts(
        model.experts,
        expert_group=expert_group,
        tensor_group=tensor_group,
    )
    model.attention = fully_shard(attention, shard_group, rank)
    model.experts = fully_shard(experts, shard_group, rank)
    return model


def sync_gradients(model: TransformerBlock, dims: ParallelDims) -> None:
    rank = get_rank()
    if not isinstance(model.attention, FSDP) or not isinstance(model.experts, FSDP):
        raise TypeError("sync_gradients expects a parallelized TransformerBlock")

    sync_replicated_gradients(
        model.attention,
        (
            dims.group(rank, "dp_replicate"),
            dims.group(rank, "cp"),
            dims.group(rank, "ep"),
        ),
    )
    sync_replicated_gradients(
        model.experts,
        (
            dims.group(rank, "dp_replicate"),
            dims.group(rank, "cp"),
        ),
    )
