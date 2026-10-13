import torch
from torch import Tensor, nn

from ..emulator import all_to_all, get_rank
from ..model import MoE
from ..parallel_dims import Group
from .tensor import parallelize_mlp


class ExpertParallelMoE(nn.Module):
    def __init__(self, module: MoE, expert_group: Group, tensor_group: Group) -> None:
        super().__init__()
        if len(module.experts) % expert_group.size:
            raise ValueError("the number of experts must be divisible by the expert parallel size")

        self.expert_group = expert_group
        self.experts_per_rank = len(module.experts) // expert_group.size
        start = expert_group.local_rank(get_rank()) * self.experts_per_rank
        self.experts = nn.ModuleList(
            parallelize_mlp(expert, tensor_group)
            for expert in module.experts[start : start + self.experts_per_rank]
        )

    def forward(self, tensor: Tensor) -> Tensor:
        shape = tensor.shape
        tokens = tensor.reshape(-1, shape[-1])
        expert_count = self.experts_per_rank * self.expert_group.size
        if tokens.shape[0] % expert_count:
            raise RuntimeError("tokens per rank must be divisible by the number of experts")

        owners = torch.arange(tokens.shape[0]) % expert_count
        destinations = owners // self.experts_per_rank
        order = torch.argsort(destinations, stable=True)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel())
        dispatched = all_to_all(tokens[order], self.expert_group)
        local_owners = all_to_all(owners[order] % self.experts_per_rank, self.expert_group)
        processed = torch.empty_like(dispatched)
        for index, expert in enumerate(self.experts):
            selected = local_owners == index
            processed[selected] = expert(dispatched[selected])
        returned = all_to_all(processed, self.expert_group)
        return returned[inverse].view(shape)


def parallelize_experts(module: MoE, *, expert_group: Group, tensor_group: Group) -> ExpertParallelMoE:
    return ExpertParallelMoE(module, expert_group, tensor_group)
