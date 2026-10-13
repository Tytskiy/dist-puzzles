import copy

import flock
import torch
import torch.nn.functional as F


def _results(actual, expected):
    assert len(actual) == len(expected), "Return one result per rank."
    for rank, (got, want) in enumerate(zip(actual, expected, strict=True)):
        torch.testing.assert_close(
            got, want, rtol=2e-5, atol=2e-5, msg=lambda msg, rank=rank: f"Rank {rank}: {msg}"
        )


def _communication(trace, collectives=None):
    assert all(record.match is not None for record in trace.p2p), "Finish every send and receive."
    if collectives is not None:
        used = {record.kind for record in trace.collectives}
        assert used <= collectives, f"Unexpected collectives: {used - collectives}."


def check_ddp_step(fn):
    for world, batch, width in ((1, 3, 2), (2, 2, 3), (4, 3, 2)):
        for seed in (0, 1, 7):
            generator = torch.Generator().manual_seed(0)
            with torch.random.fork_rng():
                torch.manual_seed(0)
                model = torch.nn.Sequential(torch.nn.Linear(width, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
            replicas = [copy.deepcopy(model) for _ in range(world)]
            optimizer = torch.optim.SGD(model.parameters(), lr=0.07)
            for _ in range(2):
                inputs = [torch.randn(batch, width, generator=generator) for _ in range(world)]
                targets = [torch.randn(batch, 2, generator=generator) for _ in range(world)]
                optimizer.zero_grad()
                F.mse_loss(model(torch.cat(inputs)), torch.cat(targets)).backward()
                optimizer.step()
                run = flock.distribute(workers=world, seed=seed, trace=True)(fn)
                results, trace = run(
                    flock.per_rank(replicas), flock.per_rank(inputs), flock.per_rank(targets), 0.07
                )
                expected = tuple(p.detach().clone() for p in model.parameters())
                _results(results, [expected] * world)
                _results(
                    [tuple(p.detach() for p in replica.parameters()) for replica in replicas],
                    [expected] * world,
                )
                _communication(trace)
    print("All checks passed.")


def _linear(fn, column):
    generator = torch.Generator().manual_seed(0)
    for world in (1, 2, 3):
        for dtype in (torch.float32, torch.float64):
            x = torch.randn(3, 4 * world, generator=generator, dtype=dtype)
            w = torch.randn(4 * world, 2 * world, generator=generator, dtype=dtype)
            b = torch.randn(2 * world, generator=generator, dtype=dtype)
            expected = x @ w + b
            for seed in (0, 1, 7):
                run = flock.distribute(workers=world, seed=seed, trace=True)(fn)
                results, trace = run(
                    x.clone() if column else flock.per_rank(x.clone().chunk(world, 1)),
                    flock.per_rank(w.clone().chunk(world, 1 if column else 0)),
                    flock.per_rank(b.clone().chunk(world)) if column else b.clone(),
                )
                _results(results, [expected] * world)
                _communication(trace)
    print("All checks passed.")


def check_columnwise_tp(fn):
    _linear(fn, column=True)


def check_rowwise_tp(fn):
    _linear(fn, column=False)


def check_tp_mlp(fn):
    generator = torch.Generator().manual_seed(0)
    for world in (1, 2, 3):
        for tokens, width, dtype in ((1, 4, torch.float32), (5, 6, torch.float64)):
            hidden = 3 * world
            x = torch.randn(tokens, width, generator=generator, dtype=dtype)
            w0 = torch.randn(width, hidden, generator=generator, dtype=dtype)
            w1 = torch.randn(hidden, width, generator=generator, dtype=dtype)
            b0 = torch.randn(hidden, generator=generator, dtype=dtype)
            b1 = torch.randn(width, generator=generator, dtype=dtype)
            expected = F.gelu(x @ w0 + b0) @ w1 + b1
            for seed in (0, 1, 7):
                run = flock.distribute(workers=world, seed=seed, trace=True)(fn)
                results, trace = run(
                    x.clone(),
                    flock.per_rank(w0.clone().chunk(world, 1)),
                    flock.per_rank(w1.clone().chunk(world)),
                    flock.per_rank(b0.clone().chunk(world)),
                    b1.clone(),
                )
                _results(results, [expected] * world)
                _communication(trace, {"all_reduce", "reduce", "broadcast"})
    print("All checks passed.")


def _attention(q, k, v):
    return torch.softmax(q @ k.mT / q.shape[-1] ** 0.5, dim=-1) @ v


def check_tp_attention(fn):
    generator = torch.Generator().manual_seed(0)
    for world in (1, 2, 4):
        for batch, sequence, dtype in ((1, 1, torch.float32), (2, 5, torch.float64)):
            heads, head_dim = 4, 2
            width = heads * head_dim
            x = torch.randn(batch, sequence, width, generator=generator, dtype=dtype)
            weights = [
                torch.randn(width, width, generator=generator, dtype=dtype) / width**0.5 for _ in range(4)
            ]
            q, k, v = [(x @ w).reshape(batch, sequence, heads, head_dim).transpose(1, 2) for w in weights[:3]]
            expected = _attention(q, k, v).transpose(1, 2).reshape(batch, sequence, width) @ weights[3]
            for seed in (0, 1, 7):
                run = flock.distribute(workers=world, seed=seed, trace=True)(fn)
                results, trace = run(
                    x.clone(),
                    *(flock.per_rank(w.clone().chunk(world, 1)) for w in weights[:3]),
                    flock.per_rank(weights[3].clone().chunk(world)),
                    heads,
                )
                _results(results, [expected] * world)
                _communication(trace, {"all_reduce", "reduce", "broadcast"})
    print("All checks passed.")


def check_expert_parallel(fn):
    generator = torch.Generator().manual_seed(0)
    for world in (1, 2, 4):
        for dtype in (torch.float32, torch.float64):
            experts, width = 2 * world, 3
            weights = torch.randn(experts, width, width, generator=generator, dtype=dtype)
            biases = torch.randn(experts, width, generator=generator, dtype=dtype)
            for pattern in ("random", "one_expert", "empty"):
                counts = [0] * world if pattern == "empty" else [r + 2 for r in range(world)]
                if world > 1:
                    counts[0] = 0
                inputs = [torch.randn(count, width, generator=generator, dtype=dtype) for count in counts]
                ids = [
                    torch.randint(experts, (count,), generator=generator)
                    if pattern == "random"
                    else torch.full((count,), experts - 1, dtype=torch.long)
                    for count in counts
                ]
                expected = [
                    F.gelu(torch.einsum("td,tde->te", x, weights[assigned]) + biases[assigned])
                    for x, assigned in zip(inputs, ids, strict=True)
                ]
                for seed in (0, 1, 7):
                    run = flock.distribute(workers=world, seed=seed, trace=True)(fn)
                    results, trace = run(
                        flock.per_rank([x.clone() for x in inputs]),
                        flock.per_rank([i.clone() for i in ids]),
                        flock.per_rank(weights.clone().chunk(world)),
                        flock.per_rank(biases.clone().chunk(world)),
                    )
                    _results(results, expected)
                    _communication(trace, {"all_to_all"})
    print("All checks passed.")


def _context_attention(fn, layout):
    generator = torch.Generator().manual_seed(0)
    worlds = (1, 2, 4) if layout == "heads" else (1, 2, 3, 4)
    for world in worlds:
        heads = 4 if layout == "heads" else 3
        for batch, local_sequence, dtype, scale in (
            (1, 1, torch.float32, 1),
            (2, 3, torch.float64, 1),
            (1, 2, torch.float32, 15),
        ):
            shape = (batch, heads, world * local_sequence, 2)
            q, k, v = [torch.randn(shape, generator=generator, dtype=dtype) for _ in range(3)]
            q, k = q * scale, k * scale
            expected = list(_attention(q, k, v).chunk(world, dim=2))
            for seed in (0, 1, 7):
                run = flock.distribute(workers=world, seed=seed, trace=True)(fn)
                results, trace = run(*(flock.per_rank(t.clone().chunk(world, 2)) for t in (q, k, v)))
                _results(results, expected)
                _communication(
                    trace, set() if layout == "ring" else {"all_to_all"} if layout == "heads" else None
                )
                if layout == "ring":
                    sends = [record.begin for record in trace.p2p if record.begin.op in {"send", "isend"}]
                    assert all(s.peer == (s.rank + 1) % world for s in sends), (
                        "Send K/V only to the right neighbour."
                    )
                    if world == 1:
                        assert not sends, "One rank already owns all K/V; no exchange is needed."
    print("All checks passed.")


def check_gathered_attention(fn):
    _context_attention(fn, "gather")


def check_ulysses(fn):
    _context_attention(fn, "heads")


def check_ring_attention(fn):
    _context_attention(fn, "ring")
