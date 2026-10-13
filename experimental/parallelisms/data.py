from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call

from ..emulator import all_gather, all_reduce
from ..parallel_dims import Group


@dataclass(frozen=True)
class _ParameterSpec:
    name: str
    shape: torch.Size
    elements: int


class FSDP(nn.Module):
    def __init__(self, module: nn.Module, group: Group, rank: int) -> None:
        super().__init__()
        object.__setattr__(self, "_wrapped_module", module)
        self.group = group

        named_parameters = list(module.named_parameters())
        if not named_parameters:
            raise ValueError("FSDP needs a module with parameters")

        self._parameter_specs = [
            _ParameterSpec(name, parameter.shape, parameter.numel()) for name, parameter in named_parameters
        ]
        flat = torch.cat([parameter.detach().flatten() for _, parameter in named_parameters])
        self._elements = flat.numel()
        padding = (-self._elements) % group.size
        if padding:
            flat = F.pad(flat, (0, padding))
        self.shard = nn.Parameter(flat.chunk(group.size)[group.local_rank(rank)].clone())

        for _, parameter in named_parameters:
            parameter.requires_grad_(False)

    def forward(self, tensor: Tensor) -> Tensor:
        flat = all_gather(self.shard, self.group)[: self._elements]
        parameters: dict[str, Tensor] = {}
        offset = 0
        for spec in self._parameter_specs:
            parameters[spec.name] = flat[offset : offset + spec.elements].view(spec.shape)
            offset += spec.elements
        return cast(Tensor, functional_call(self._wrapped_module, parameters, (tensor,)))


def fully_shard(module: nn.Module, group: Group, rank: int) -> FSDP:
    return FSDP(module, group, rank)


def sync_replicated_gradients(module: nn.Module, groups: Iterable[Group]) -> None:
    active_groups = [group for group in groups if group.size > 1]
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad
        for group in active_groups:
            gradient = all_reduce(gradient, group)
        parameter.grad = gradient
