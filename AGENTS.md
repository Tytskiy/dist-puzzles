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

- `puzzles/` — notebooks and exercise specs; this is the main project surface
  for now.
- `src/dist_puzzles/` — tiny shared notebook helpers: `check`, `check_comm`,
  and `comm_report`.
- `pyproject.toml` — dependency, package, and tool configuration.
- `justfile` — run `just check`, `just test`, `just fix`.
- Do not add `tests/` yet. Add pytest tests only when the user explicitly
  decides the project needs them.

## Puzzle authoring

- If the user asks a question, answer it first and do not take action unless
  they explicitly ask for an edit, command, or implementation.
- Move at the user's pace. If the user asks for one puzzle, task, or edit, do
  only that work and stop; do not proactively build future tiers material.
- The user owns the teaching narrative, video flow, and storytelling. Do not add
  broad explanations, lesson scripts, or narrative framing around puzzles.
- Puzzle text may include a small neutral description when it helps identify the
  task, expected output, or grading target, but keep it minimal and practical.

## Tooling

- `uv` for dependency management
- `ruff` for linting and formatting
- There is no pytest surface in this repo for now.
- Run `just check && just test` after changes.

## Relationship to flock

- **flock is the runtime.** Scheduling, p2p, built-in collectives, deadlock
  diagnostics, and `@flock.distribute` live in the sibling `flock` package.
  Do not reimplement them here.
- **dist-puzzles is the course layer.** PyTorch tensors, tensor-specific
  collectives, mesh/placement abstractions, and puzzle grading.
- Import flock for communication (`flock.isend`, `flock.all_gather`, …) and
  `flock.torch` for tensor communication. Keep puzzle-specific code in
  notebooks unless the user asks for shared modules.
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
