from collections import Counter

import flock
import torch

from dist_puzzles import check


def _check_collective(fn, operation, topology, worlds=(1, 2, 3, 5)):
    generator = torch.Generator().manual_seed(0)
    for world in worlds:
        for chunk_size, dtype in ((1, torch.int64), (3, torch.float32), (2, torch.float64)):
            size = chunk_size if operation == "gather" else world * chunk_size
            inputs = [torch.randint(-20, 21, (size,), generator=generator).to(dtype) for _ in range(world)]
            if dtype.is_floating_point:
                inputs = [x / 8 for x in inputs]
            total = torch.cat(inputs) if operation == "gather" else torch.stack(inputs).sum(dim=0)
            expected = list(total.chunk(world)) if operation == "scatter" else [total] * world
            for seed in (0, 1, 7):
                run = flock.distribute(workers=world, seed=seed, trace=True)(fn)
                results, trace = run(flock.per_rank([x.clone() for x in inputs]))
                check(results, expected)
                for rank, (result, target) in enumerate(zip(results, expected, strict=True)):
                    assert result.shape == target.shape, f"Rank {rank}: return shape {tuple(target.shape)}."
                    assert result.dtype == dtype, f"Rank {rank}: preserve the input dtype {dtype}."
                assert not trace.collectives, "Build the collective using point-to-point communication."
                assert all(r.match is not None for r in trace.p2p), "Finish every send and receive."
                sends = [r.begin for r in trace.p2p if r.begin.op in {"send", "isend"}]
                if world == 1:
                    assert not sends, "With one rank, return the local result without communication."
                    continue
                if topology == "root":
                    assert all(s.rank == 0 or s.peer == 0 for s in sends), "Communicate through rank 0."
                    continue
                counts = Counter(s.rank for s in sends)
                if topology == "ring":
                    assert all(s.peer == (s.rank + 1) % world for s in sends), (
                        "Send only to the next ring rank."
                    )
                    passes = 2 if operation == "reduce" else 1
                    assert all(count <= passes * (world - 1) for count in counts.values()), (
                        f"Use at most {passes * (world - 1)} sends per rank."
                    )
                    chunk_bytes = chunk_size * inputs[0].element_size()
                    assert all(s.nbytes == chunk_bytes for s in sends), "Send one tensor chunk per message."
                else:
                    assert all((delta := s.rank ^ s.peer) > 0 and delta & (delta - 1) == 0 for s in sends), (
                        "Exchange with partners whose rank differs in one binary digit."
                    )
                    assert all(count <= world.bit_length() - 1 for count in counts.values()), (
                        "Use at most log2(world_size) sends per rank."
                    )
                    budget = (world - 1) * inputs[0].numel() * inputs[0].element_size()
                    assert all(amount <= budget for amount in trace.summary().bytes_sent), (
                        "Send at most world_size - 1 input chunks per rank in total."
                    )
    print("All checks passed.")


def check_root_all_reduce(fn):
    _check_collective(fn, "reduce", "root")


def check_ring_all_gather(fn):
    _check_collective(fn, "gather", "ring")


def check_ring_reduce_scatter(fn):
    _check_collective(fn, "scatter", "ring")


def check_ring_all_reduce(fn):
    _check_collective(fn, "reduce", "ring")


def check_tree_all_gather(fn):
    _check_collective(fn, "gather", "doubling", worlds=(1, 2, 4, 8))
