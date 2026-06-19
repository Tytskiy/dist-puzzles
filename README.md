# dist-puzzles

Learn distributed training from scratch — tensors, collectives, and parallelisms.

Each puzzle is a spec, an expected output, and a body to fill in. Programs run on
[flock](https://github.com/tytskiy/flock): a single-process emulator where every
rank is a coroutine and communication goes through in-memory mailboxes. Same mental
model as `torch.distributed`, runnable on a laptop with real CPU tensors and autograd.

```python
import flock
import torch
from dist_puzzles.grading import check_puzzle

@flock.distribute(workers=4)
async def sum_ranks():
    t = torch.tensor([flock.get_rank()], dtype=torch.float32)
    return (await flock.all_reduce(t, "sum").wait()).item()

check_puzzle(sum_ranks, [6.0, 6.0, 6.0, 6.0], workers=4)
```

## flock vs dist-puzzles

| | flock | dist-puzzles |
|---|-------|--------------|
| Role | Emulator / runtime | Course + tensor layer |
| Dependencies | Pure Python | PyTorch + flock |
| Collectives | Built-in, value-typed | Tensor-aware reimplementations |
| Students build | — | `DeviceMesh`, `DTensor`, DDP, FSDP, TP, … |

flock stays minimal and dependency-free. dist-puzzles adds torch payloads, puzzle
grading, and the higher-level abstractions students implement tier by tier.

## Project layout

```
src/dist_puzzles/
  grading/         # puzzle check helpers
puzzles/           # notebooks and tiered exercises
tests/             # pytest, mirroring package layout
```

See [IDEA.md](IDEA.md) for the full course arc.

## Development

Requires [uv](https://docs.astral.sh/uv/). flock is linked as an editable path
dependency (`../flock` in `pyproject.toml`).

```bash
uv sync
just check   # ruff + mypy
just test    # pytest
just fix     # auto-format
```

## More detail

- [IDEA.md](IDEA.md) — course tiers, what students build, redistribution as collectives
- [flock README](../flock/README.md) — emulator API, scheduling, deadlock diagnostics
