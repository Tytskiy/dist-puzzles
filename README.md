# Distributed Training Puzzles

> [!WARNING]
> **ALPHA — WORK IN PROGRESS.**
> This course is unfinished and actively changing. Lessons, exercises, solutions,
> and checkers may contain mistakes or change without notice. Expect rough edges;
> this is not a stable release.

Learn distributed programming and training through small Python puzzles, using
[flock](https://github.com/Tytskiy/flock) to run simulated workers locally and
real CPU PyTorch tensors for the training examples. No GPUs or cluster required.

## Getting started

Install Python 3.12+ and [uv](https://docs.astral.sh/uv/), then clone both projects
next to each other. The current development setup requires `../flock`:

```sh
git clone https://github.com/Tytskiy/flock.git
git clone https://github.com/Tytskiy/dist-puzzles.git
cd dist-puzzles
uv sync
uv run --with jupyterlab jupyter lab
```

Open the notebooks in order:

1. [Introduction](puzzles/00_intro.ipynb)
2. [Point-to-point communication](puzzles/01_p2p.ipynb)
3. [Building collectives](puzzles/02_p2p_collectives.ipynb)
4. [Tensor, context, and expert parallelism](puzzles/03_tp_cp_ep.ipynb)
5. [Pipeline parallelism](puzzles/05_pipeline_parallel.ipynb)

The notebooks currently include implementations alongside exercises and checks.
Flock simulates communication and scheduling; its traces are not GPU or network
performance benchmarks.

## Development

With [just](https://github.com/casey/just) installed:

```sh
just check && just test
```

`just check` runs lint checks. `just test` is currently a placeholder; it does not
execute the notebooks. Run their cells to exercise the examples and checkers.
