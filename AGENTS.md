# Agent guidelines for dist-puzzles

## Comments

Do not write comments that narrate what code does. The code speaks for itself.

A comment is justified only when the code alone cannot convey *why*
something is done, a non-obvious trade-off, or a subtle constraint.

## Style

- Python 3.12+. Use PEP 695 generics (`class Foo[T]:`, `def bar[T]():`)
  instead of `TypeVar`.
- `@dataclass` for data, plain classes for bases with no fields.
- `ClassVar` for class-level constants on dataclasses.
- `from __future__ import annotations` only when a module needs forward
  references.
- No docstrings on classes or functions whose name and signature already
  explain what they do.

## Project structure

- `src/dist_puzzles/` — library source (shared helpers, not student solutions)
- `src/dist_puzzles/grading/` — `check_puzzle` and related test helpers
- `puzzles/` — tiered notebooks and exercise specs
- `tests/` — pytest mirroring package layout; `tests/typing/` for static checks
- `justfile` — run `just check` (ruff + mypy), `just test`, `just fix`

## Tooling

- `uv` for dependency management
- `ruff` for linting and formatting
- `mypy` for type checking (`src` + `tests/typing`, strict-ish)
- `pytest` for tests
- Run `just check && just test` after changes

## Relationship to flock

- **flock is the runtime.** Scheduling, p2p, built-in collectives, deadlock
  diagnostics, and `@flock.distribute` live in the sibling `flock` package.
  Do not reimplement them here.
- **dist-puzzles is the course layer.** PyTorch tensors, tensor-specific
  collectives, mesh/placement abstractions, and puzzle grading.
- Import flock for communication (`flock.isend`, `flock.all_gather`, …). Add
  puzzle-specific code under `puzzles/` or new `src/dist_puzzles/` modules as
  tiers need them.
- Keep flock dependency-free: never add torch imports to flock from this repo.

## Design principles

- Payloads are real CPU `torch.Tensor` values with autograd where puzzles need it.
- Tensor collectives wrap flock's value-typed collectives with shape/dtype-aware
  semantics (concatenate on gather, elementwise reduce, etc.).
- High-level parallel APIs (`DeviceMesh`, `DTensor`, `redistribute`) are student
  puzzles — add them when a tier needs them, not ahead of time.
- Every redistribution maps to a collective:
  `Shard → Replicate` = `all_gather`, `Partial → Replicate` = `all_reduce`,
  `Shard(i) → Shard(j)` = `all_to_all`.
- Error messages should be clear and actionable, aimed at someone learning
  distributed training.
- Puzzle grading uses deterministic seeds (`Random(seed=0)`) unless a puzzle
  explicitly tests scheduling sensitivity.
