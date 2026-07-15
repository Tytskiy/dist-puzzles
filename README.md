# dist-puzzles

Learn distributed training from scratch — tensors, collectives, and parallelisms.

Each puzzle is a spec, an expected output, and a body to fill in. Programs run on
[flock](https://github.com/Tytskiy/flock): a single-process emulator where every
rank is a coroutine and communication goes through in-memory mailboxes. Same mental
model as `torch.distributed`, runnable on a laptop with real CPU tensors and autograd.

```python
import flock
import flock.torch as ft
import torch

from dist_puzzles import check


@flock.distribute(workers=4)
async def sum_ranks():
    t = torch.tensor([flock.get_rank()], dtype=torch.float32)
    return await ft.all_reduce(t).wait()


check(sum_ranks(), [torch.tensor([6.0])] * 4)
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
  checks.py         # result checks
  comm.py           # communication checks and reports
puzzles/           # notebooks and tiered exercises
```

See [IDEA.md](IDEA.md) for the full course arc.

## Development

Requires [uv](https://docs.astral.sh/uv/). flock is linked as an editable path
dependency (`../flock` in `pyproject.toml`).

```bash
uv sync
just check   # ruff
just test    # notebook test placeholder
just fix     # auto-format
```

## More detail

- [IDEA.md](IDEA.md) — course tiers, what students build, redistribution as collectives
- [flock README](../flock/README.md) — emulator API, scheduling, deadlock diagnostics
