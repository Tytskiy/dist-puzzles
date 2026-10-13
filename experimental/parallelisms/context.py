from math import sqrt

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..emulator import all_gather, all_reduce, get_rank
from ..model import Attention
from ..parallel_dims import Group


class ParallelAttention(nn.Module):
    def __init__(self, module: Attention, tensor_group: Group, context_group: Group) -> None:
        super().__init__()
        if module.heads % tensor_group.size:
            raise ValueError("attention heads must divide evenly across tensor parallel ranks")

        hidden_size = module.qkv.in_features
        local_hidden = hidden_size // tensor_group.size
        local_rank = tensor_group.local_rank(get_rank())
        start = local_rank * local_hidden
        stop = start + local_hidden
        q, k, v = module.qkv.weight.detach().chunk(3, dim=0)
        q_bias, k_bias, v_bias = module.qkv.bias.detach().chunk(3)

        self.local_heads = module.heads // tensor_group.size
        self.head_size = module.head_size
        self.tensor_group = tensor_group
        self.context_group = context_group
        self.qkv = nn.Parameter(torch.cat((q[start:stop], k[start:stop], v[start:stop])))
        self.qkv_bias = nn.Parameter(torch.cat((q_bias[start:stop], k_bias[start:stop], v_bias[start:stop])))
        self.projection = nn.Parameter(module.projection.weight.detach()[:, start:stop].clone())

    def forward(self, tensor: Tensor) -> Tensor:
        batch, sequence, _ = tensor.shape
        q, k, v = F.linear(tensor, self.qkv, self.qkv_bias).chunk(3, dim=-1)
        q = q.view(batch, sequence, self.local_heads, self.head_size).transpose(1, 2)
        k = k.view(batch, sequence, self.local_heads, self.head_size)
        v = v.view(batch, sequence, self.local_heads, self.head_size)
        k, v = all_gather(torch.cat((k, v), dim=-1), self.context_group, dim=1).chunk(2, dim=-1)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / sqrt(self.head_size)
        local = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(batch, sequence, -1)
        return all_reduce(F.linear(local, self.projection), self.tensor_group)


def parallelize_attention(
    module: Attention,
    *,
    tensor_group: Group,
    context_group: Group,
) -> ParallelAttention:
    return ParallelAttention(module, tensor_group, context_group)
