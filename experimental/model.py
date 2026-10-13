from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 8
    intermediate_size: int = 16
    heads: int = 4
    experts: int = 2
    seed: int = 0


def _linear(in_features: int, out_features: int, *, bias: bool, generator: torch.Generator) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=bias)
    scale = in_features**-0.5
    with torch.no_grad():
        layer.weight.copy_(torch.randn(out_features, in_features, generator=generator) * scale)
        if layer.bias is not None:
            layer.bias.copy_(torch.randn(out_features, generator=generator) * scale)
    return layer


class MLP(nn.Module):
    def __init__(self, config: ModelConfig, *, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.up = _linear(config.hidden_size, config.intermediate_size, bias=True, generator=generator)
        self.down = _linear(config.intermediate_size, config.hidden_size, bias=False, generator=generator)

    def forward(self, tensor: Tensor) -> Tensor:
        return self.down(F.gelu(self.up(tensor)))


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.hidden_size % config.heads:
            raise ValueError("attention heads must divide the hidden size")
        self.heads = config.heads
        self.head_size = config.hidden_size // config.heads
        generator = torch.Generator().manual_seed(config.seed + 1)
        self.qkv = _linear(config.hidden_size, 3 * config.hidden_size, bias=True, generator=generator)
        self.projection = _linear(config.hidden_size, config.hidden_size, bias=False, generator=generator)

    def forward(self, tensor: Tensor) -> Tensor:
        batch, sequence, _ = tensor.shape
        q, k, v = self.qkv(tensor).chunk(3, dim=-1)
        q = q.view(batch, sequence, self.heads, self.head_size).transpose(1, 2)
        k = k.view(batch, sequence, self.heads, self.head_size).transpose(1, 2)
        v = v.view(batch, sequence, self.heads, self.head_size).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / sqrt(self.head_size)
        output = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(batch, sequence, -1)
        return self.projection(output)


class MoE(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.experts < 1:
            raise ValueError("experts must be positive")
        self.experts = nn.ModuleList(
            MLP(config, seed=config.seed + 2 + expert * 1_000) for expert in range(config.experts)
        )

    def forward(self, tensor: Tensor) -> Tensor:
        shape = tensor.shape
        tokens = tensor.reshape(-1, shape[-1])
        owners = torch.arange(tokens.shape[0]) % len(self.experts)
        output = torch.empty_like(tokens)
        for index, expert in enumerate(self.experts):
            selected = owners == index
            output[selected] = expert(tokens[selected])
        return output.view(shape)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.experts = MoE(config)

    def forward(self, tensor: Tensor) -> Tensor:
        tensor = tensor + self.attention(tensor)
        return tensor + self.experts(F.gelu(tensor))
