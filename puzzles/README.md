# Puzzles

Tiered exercises live here as notebooks or Python modules. Each puzzle follows the
same shape:

1. **Spec** — what to implement and the expected per-rank output
2. **Skeleton** — function body with `...` or `raise NotImplementedError`
3. **Check** — `dist_puzzles.grading.check_puzzle(fn, expected, workers=N)`

## Tiers

| Tier | Focus | Location |
|------|-------|----------|
| 0 | Mental model (values) | `tier0/` |
| 1 | P2P → collectives → algorithms | `tier1/` |
| 2 | Tensors, DeviceMesh, DTensor | `tier2/` |
| 3 | Data parallel (DDP, ZeRO) | `tier3/` |
| 4 | Model parallel (FSDP, TP, CP, EP, PP) | `tier4/` |

Start with `tier0/intro.ipynb` (ported from flock's early exercises).
