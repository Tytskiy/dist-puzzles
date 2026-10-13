import copy

import flock
import torch

from dist_puzzles import check_comm


def _make_stages(world_size: int, width: int) -> list[torch.nn.Module]:
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")

    stages = []
    for rank in range(world_size):
        layers: list[torch.nn.Module] = [torch.nn.Linear(width, width)]
        if rank < world_size - 1:
            layers.append(torch.nn.GELU())
        stages.append(torch.nn.Sequential(*layers))
    return stages


def _expected_1f1b_events(
    rank: int,
    world_size: int,
    microbatch_count: int,
) -> list[tuple[str, int]]:
    warmup = min(world_size - rank - 1, microbatch_count)
    remaining = microbatch_count - warmup
    events = [("F", microbatch) for microbatch in range(warmup)]
    for microbatch in range(remaining):
        events.append(("F", microbatch + warmup))
        events.append(("B", microbatch))
    events.extend(("B", microbatch) for microbatch in range(remaining, microbatch_count))
    return events


def _training_case(fn, world_size: int, microbatch_count: int, seed: int) -> flock.Tracer:
    width = 8
    microbatch_size = 2
    learning_rate = 0.05

    torch.manual_seed(1_000 + world_size)
    initial_stages = _make_stages(world_size, width)
    pipeline_stages = copy.deepcopy(initial_stages)
    reference_stages = copy.deepcopy(initial_stages)

    generator = torch.Generator().manual_seed(2_000 + world_size)
    batch_size = microbatch_count * microbatch_size
    X = torch.randn(batch_size, width, generator=generator)
    target = torch.randn(batch_size, width, generator=generator)

    distributed = flock.distribute(
        workers=world_size,
        seed=seed,
        trace=True,
    )(fn)
    results, trace = distributed(
        flock.per_rank(pipeline_stages),
        X,
        target,
        microbatch_count,
        learning_rate,
    )

    reference = torch.nn.Sequential(*reference_stages)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=learning_rate)
    reference_optimizer.zero_grad(set_to_none=True)
    reference_output = reference(X)
    torch.nn.functional.mse_loss(reference_output, target).backward()
    reference_optimizer.step()

    for rank, (events, parameters, output) in enumerate(results):
        assert events == _expected_1f1b_events(rank, world_size, microbatch_count), (
            f"Rank {rank}: use warmup, oldest-first 1F1B, then cooldown."
        )
        for actual, expected in zip(
            parameters,
            reference_stages[rank].parameters(),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        for actual, expected in zip(
            pipeline_stages[rank].parameters(), reference_stages[rank].parameters(), strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        if rank == world_size - 1:
            assert output is not None
            torch.testing.assert_close(output, reference_output.detach())
        else:
            assert output is None

    links = {link for rank in range(world_size - 1) for link in ((rank, rank + 1), (rank + 1, rank))}
    check_comm(
        trace,
        allowed_links=links,
        required_links=links,
    )
    assert not trace.collectives, "Use point-to-point messages between adjacent stages."
    assert all(record.match is not None for record in trace.p2p), "Finish every exchange."
    return trace


def check_pipeline_1f1b(fn):
    for seed in (0, 1, 7):
        for depth, microbatches in ((1, 3), (2, 5), (4, 8), (6, 3), (4, 1)):
            with torch.random.fork_rng():
                _training_case(fn, depth, microbatches, seed)
    print("All checks passed.")


def check_pipeline_forward(fn):
    generator = torch.Generator().manual_seed(0)
    for depth in (1, 2, 4):
        for microbatches in (1, 3, 6):
            for seed in (0, 1, 7):
                with torch.random.fork_rng():
                    torch.manual_seed(0)
                    stages = _make_stages(depth, width=4)
                x = torch.randn(12, 4, generator=generator)
                expected = torch.nn.Sequential(*copy.deepcopy(stages))(x)
                run = flock.distribute(workers=depth, seed=seed, trace=True)(fn)
                results, trace = run(
                    flock.per_rank(stages), flock.per_rank([x] + [None] * (depth - 1)), microbatches
                )
                assert all(value is None for value in results[:-1]), "Only the last stage returns output."
                torch.testing.assert_close(results[-1], expected)
                assert not trace.collectives, "Use point-to-point messages between adjacent stages."
                assert all(record.match is not None for record in trace.p2p), "Finish every exchange."
                links = {(r, r + 1) for r in range(depth - 1)}
                check_comm(trace, allowed_links=links, required_links=links)
    print("All checks passed.")
