import torch.nn.functional as F
from torch import Tensor, nn

from ..emulator import all_reduce, get_rank
from ..model import MLP
from ..parallel_dims import Group


class TensorParallelMLP(nn.Module):
    def __init__(self, module: MLP, group: Group) -> None:
        super().__init__()
        if module.up.out_features % group.size:
            raise ValueError("intermediate size must be divisible by tensor parallel size")

        local_rank = group.local_rank(get_rank())
        self.group = group
        self.up = nn.Parameter(module.up.weight.detach().chunk(group.size, dim=0)[local_rank].clone())
        self.up_bias = nn.Parameter(module.up.bias.detach().chunk(group.size)[local_rank].clone())
        self.down = nn.Parameter(module.down.weight.detach().chunk(group.size, dim=1)[local_rank].clone())

    def forward(self, tensor: Tensor) -> Tensor:
        hidden = F.gelu(F.linear(tensor, self.up, self.up_bias))
        partial = F.linear(hidden, self.down)
        return all_reduce(partial, self.group)


def parallelize_mlp(module: MLP, group: Group) -> TensorParallelMLP:
    return TensorParallelMLP(module, group)
