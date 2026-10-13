# Parallel training experiment

The capstone starts with an ordinary single-device model in [`model.py`](model.py). Parallelism is applied
afterward, so every distributed change has a visible purpose:

- [`parallel_dims.py`](parallel_dims.py) factors the world into named parallel axes.
- [`parallelisms/`](parallelisms) contains the parameter layouts and communication for DP, FSDP, CP, TP,
  EP, and PP.
- [`parallelize.py`](parallelize.py) composes those techniques for the model.
- [`train.py`](train.py) keeps model construction, parallelization, and the training step visible.

The `dp_replicate` axis gives different data to replicas and synchronizes their gradients. The `dp_shard`
axis holds FSDP parameter shards that cooperate on the same input. The remaining axes divide sequence,
tensor, expert, and pipeline work.

[`emulator.py`](emulator.py) is the small public runtime surface. The `_emulator` package contains the
greenlet scheduler, PyTorch thread-local-state handling, and the global autograd handoff. These mechanisms
make rank-local code resumable; they are not part of the parallel algorithms students need to learn.

The emulator preserves tensor values, gradients, ownership, and communication dependencies. It does not
model execution time or communication overlap.

Run the combined 64-rank example with:

```bash
python -m experimental.train
```
