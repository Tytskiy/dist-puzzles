# dist-puzzles

A "build distributed training from scratch" course, in the spirit of the Tensor /
GPU / Triton Puzzles. Each puzzle is a spec + expected output + a body to fill,
graded automatically on top of [flock](../flock).

## Substrate

- **Runtime: flock.** Single-process emulator; ranks are coroutines; communication
  through in-memory mailboxes. Deterministic scheduling and deadlock diagnostics
  let students *see* interleavings and hangs that real frameworks hide.
- **Tensors & autograd: real (CPU) `torch`.** Payloads are torch tensors; forward
  and backward use torch autograd. Lessons transfer to actual `torch.distributed`.

## What flock provides vs. what students build

flock is the substrate: p2p, built-in collectives, scheduler, groups, tags.
dist-puzzles adds the course layer on top.

**Students build (the puzzles):**

- Tensor collectives with proper shape semantics (`all_gather_tensor`, …)
- `DeviceMesh` — an n-d reshape of the world over `flock.new_group`
- `Placements` (`Shard` / `Replicate` / `Partial`)
- `DTensor` + `redistribute` (each placement change is a collective)
- Parallelisms: DDP, FSDP, TP, CP, EP, PP

Shipping `DeviceMesh` or `redistribute` as finished library code would hand
students the punchline. Add abstractions tier by tier as puzzles require them.

## Two ideas the course is built around

1. **Collectives are p2p algorithms.** Implement `all_reduce` (ring vs recursive
   halving/doubling), `broadcast` (flat vs tree), etc., from `send`/`recv`, and
   compare them by message count / rounds / bytes.
2. **Redistribution is collectives.** Once DTensor exists, every placement change is
   a collective:
   - `Shard → Replicate` = `all_gather`
   - `Replicate → Shard` = local slice (no comm)
   - `Partial → Replicate` = `all_reduce`
   - `Shard(i) → Shard(j)` = `all_to_all`

## Tiers

0. **Mental model** — ranks, the scheduler, deadlocks (plain Python values in flock).
1. **P2P → collectives → algorithms** — semantics, then implement collectives from
   p2p (ring/tree), with a trace/cost model to compare them.
2. **Tensors + autograd → DeviceMesh / Placements / DTensor** — `redistribute` as
   collectives; DeviceMesh is an n-d reshape of the world handing back a `Group`
   per dim.
3. **Data parallel** — DDP (all_reduce grads), bucketing + comm/compute overlap,
   then ZeRO 1/2/3 as the ramp into sharding.
4. **Model parallelisms** — FSDP, TP, CP, EP, PP, and a 2D/3D composition capstone
   (TP × DP × PP).

## Topics not to forget

- Comm/compute overlap & prefetch (DDP buckets, FSDP all-gather prefetch) — built on
  flock's "post early, await late" + `work.is_completed()`.
- Gradient accumulation / micro-batching (prereq for PP).
- Sharded checkpoint save/load (DTensor state dict).
- Numerics & debugging: all_reduce float ordering, shape/rank-mismatch diagnostics.
