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

- `puzzles/` — lesson notebooks, exercise specs, implementations, examples, and
  checker calls. Keep everything together here while the course is being refactored.
  Do not recreate separate solution notebooks until the user asks.
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
- Follow the user's teaching direction and scope. Write narrative for the requested
  lesson or puzzle; leave video flow and broader course storytelling to the user.
- Keep runnable examples explicit; do not hide launching workers behind a problem framework.
- Use plain Python values and `flock` for P2P exercises. Introduce tensors only where
  tensor computation is the subject.
- Students should only need to import checkers from lesson modules; avoid public
  enums, constants, or wrappers for ordinary values. Keep supplied local logic simple.
- Check results and essential communication properties across varied inputs and schedules.
  Accept different valid protocols, message formats, and operation orders. Do not enforce
  a reference algorithm through exact trace sequences or byte counts without a task-level reason.
- Preserve student solutions when revising a notebook.
- Keep descriptions, implementations, runnable examples, checker calls, and
  visualizations consistent within each puzzle notebook.
- Use `BACKLOG.md` for concrete future features and course additions, not review reminders,
  editing plans, discussion agendas, or the user's next steps. Discuss success rewards before adding them.

## Puzzle narrative

- Assume Python fluency and a learner new to distributed programming. Spend words
  on communication, ordering, ownership, and completion; skip ordinary Python walkthroughs.
- Build around a concrete goal: what each rank knows initially and what the program
  must accomplish. Lesson introductions establish context and motivation, not a task inventory.
- Choose examples and their order before polishing prose. Reuse familiar values,
  roles, and operations while changing one important constraint at a time.
- When the narrative feels ambiguous, thin, or disjointed, check for a missing
  learning step. Proactively suggest a lesson or intermediate puzzle when it would
  close that gap. Name the gap, where the addition belongs, and what it would teach.
  Suggest the smallest useful addition; create it only when the user asks.
- Introduce a concept or API when an example makes its purpose visible. Supply the
  API knowledge needed for a puzzle before asking students to reason with it.
- Make explanations grow out of what the reader has just seen: an unmet goal,
  a consequence, or a useful change to the example. Give new topics a clear reason
  to appear. Keep transitions natural and conversational; avoid abrupt definitions
  and disconnected asides, without padding the text with empty signposting.
- Explain concrete consequences: which rank waits, which can proceed, what information
  is missing, and how completion is recognized. For a likely misconception, a small
  failing program and its execution state can show where the expectation breaks.
- Use small examples with visible results, traces, or something specific to predict.
  Make the code demonstrate the point; avoid paraphrasing it both before and after.
- Match the structure to the lesson. A new primitive may need a motivating problem;
  an extension may need only one changed constraint; a synthesis puzzle may need only
  roles and a contract. Do not force every lesson through the same narrative template.
- Keep the desired result, inputs, outputs, edge cases, and essential communication
  constraints easy to find. For a dense task, put a revealing example before detailed
  rules and group those rules by purpose.
- Separate teaching the tools from solving the puzzle. An introduction can work through
  an exchange; an exercise should leave its intended protocol, ordering, or termination
  decisions to the learner. Keep hints short and optional.
- Use a plain, slightly informal voice and consistent vocabulary. Themes should clarify
  roles without adding translation work or unrelated rules. Explain useful reasons for
  choices; let occasional humor or reassurance respond to a specific situation.
- Edit each sentence by asking what difficulty, expectation, or change it helps the
  reader understand. Cut empty transitions, staged jokes, and redundant recaps. Allow
  examples to end without commentary, and vary pacing with the difficulty of the material.

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

- Tensor lessons use real CPU `torch.Tensor` values with autograd where needed; P2P lessons can use Python values.
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
