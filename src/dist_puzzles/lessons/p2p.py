import random
from collections import Counter
from unittest.mock import patch

import flock

from dist_puzzles import check


def _life_neighbors(rank: int, shape: tuple[int, int]) -> list[int]:
    height, width = shape
    if height < 3 or width < 3:
        raise ValueError("The board must have at least three rows and three columns.")
    if not 0 <= rank < height * width:
        raise ValueError(f"Rank {rank} is outside the {height} × {width} board.")
    row, col = divmod(rank, width)
    return [
        ((row + dy) % height) * width + (col + dx) % width
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy, dx) != (0, 0)
    ]


def _run(fn, inputs, expected, seed, *args):
    run = flock.distribute(workers=len(inputs), seed=seed, trace=True)(fn)
    results, trace = run(flock.per_rank(inputs), *args)
    check(results, expected)
    return trace


def _communication(trace, routes, operations):
    assert not trace.collectives, "Use point-to-point operations only; a collective was called."
    records = trace.p2p
    assert all(r.begin.op in operations for r in records), f"Use only {', '.join(sorted(operations))}."
    assert all(r.match is not None for r in records), "Every send and receive must have a matching partner."
    sends = [r.begin for r in records if r.begin.op in {"send", "isend"}]
    actual = Counter((r.rank, r.peer) for r in sends)
    assert actual == Counter(routes), f"Expected message counts {dict(routes)}, got {dict(actual)}."


def _p2p_rules(trace, allowed_links, required_links=()):
    assert not trace.collectives, "Use point-to-point communication for this task."
    assert all(r.match is not None for r in trace.p2p), (
        "Finish all communication; some messages are unmatched."
    )
    links = {(r.begin.rank, r.begin.peer) for r in trace.p2p if r.begin.op in {"send", "isend"}}
    assert links <= allowed_links, f"Unexpected communication links: {links - allowed_links}."
    assert set(required_links) <= links, "Some required participants did not communicate."


def check_ring_shift(fn):
    for seed in (0, 1, 7):
        rng = random.Random(seed)
        for world in (2, 3, 4, 7):
            inputs = rng.sample(range(-1000, 1000), world)
            expected = inputs[-1:] + inputs[:-1]
            trace = _run(fn, inputs, expected, seed)
            links = {(rank, (rank + 1) % world) for rank in range(world)}
            _p2p_rules(trace, links, links)
    print("All checks passed.")


def check_collect_reports(fn):
    for seed in (0, 1, 7):
        rng = random.Random(seed)
        for world in (2, 3, 7):
            inputs = [None] + rng.choices(range(-3, 4), k=world - 1)
            expected = [dict(enumerate(inputs[1:], 1))] + [None] * (world - 1)
            trace = _run(fn, inputs, expected, seed)
            links = {(rank, 0) for rank in range(1, world)}
            _p2p_rules(trace, links, links)
    print("All checks passed.")


def _selective_case(fn, order, messages, seed):
    sent = []
    send, isend = flock.send, flock.isend

    def record(dst, value, tag):
        rank = flock.get_rank()
        assert type(tag) is int and tag >= 0, "Use positive result tags or tag 0 for control messages."
        if tag > 0:
            assert rank != 0 and dst == 0, "Results go from workers to rank 0."
            assert type(value) is int, "Send each integer result unchanged."
        sent.append((rank, tag, value))

    async def observed_send(dst, value, tag=0):
        record(dst, value, tag)
        await send(dst, value, tag=tag)

    def observed_isend(dst, value, tag=0):
        record(dst, value, tag)
        return isend(dst, value, tag=tag)

    world = len(messages) + 1
    run = flock.distribute(workers=world, seed=seed, trace=True)(fn)
    with patch.object(flock, "send", observed_send), patch.object(flock, "isend", observed_isend):
        results, trace = run(
            flock.per_rank([list(order)] + [None] * (world - 1)),
            flock.per_rank([None] + [list(batch) for batch in messages]),
        )
    links = {(rank, 0) for rank in range(1, world)} | {(0, rank) for rank in range(1, world)}
    _p2p_rules(trace, links)
    expected = Counter((rank, tag, value) for rank, batch in enumerate(messages, 1) for tag, value in batch)
    actual = Counter((rank, tag, value) for rank, tag, value in sent if tag > 0)
    assert actual == expected, "Send each supplied result exactly once, unchanged."
    records = [r for r in trace.p2p if r.begin.op in {"send", "isend"}]
    assert len(records) == len(sent), "Use flock.send or flock.isend for message delivery."
    deliveries = sorted(
        (record.match.step, tag, value)
        for record, (_, tag, value) in zip(records, sent, strict=True)
        if tag > 0
    )
    requested = deliveries[:len(order)]
    assert [tag for _, tag, _ in requested] == order, (
        "Receive result tags in the requested order before accepting any remaining results."
    )
    check(results, [[value for _, _, value in requested]] + [None] * (world - 1))


def check_receive_selectively(fn):
    cases = [
        ([], [[]]),
        ([], [[], [], []]),
        ([], [[(3, 8), (1, -2)], []]),
        ([2, 1], [[(1, 10), (2, 20)], [(3, 30)], []]),
        ([1], [[(1, 7), (1, 8), (2, 9)], [(1, 7)]]),
        ([2, 1], [[(1, 10), (2, 20)]]),
        ([2, 1, 2], [[(1, -4), (2, 0)], [(2, 0)], []]),
        ([1, 3, 1, 2], [[(2, 7), (1, 8)], [], [(1, 9), (3, -2)]]),
    ]
    rng = random.Random(0)
    for world in (2, 4, 7):
        order = [rng.randint(1, 4) for _ in range(16)]
        messages = [[] for _ in range(world - 1)]
        for tag in order:
            messages[rng.randrange(world - 1)].append((tag, rng.randint(-10, 10)))
        for _ in range(8):
            messages[rng.randrange(world - 1)].append((rng.randint(1, 6), rng.randint(-10, 10)))
        for batch in messages:
            rng.shuffle(batch)
        cases.append((order, messages))
    for order, messages in cases:
        for seed in (0, 1, 7):
            _selective_case(fn, order, messages, seed)
    print("All checks passed.")


def check_task_queue(fn):
    for seed in (0, 1, 7):
        rng = random.Random(seed)
        for world, count in ((2, 0), (4, 0), (5, 2), (3, 2), (4, 9)):
            jobs = rng.choices(range(-20, 20), k=count)
            calls = []
            offset = seed + 17

            def compute(job, calls=calls, offset=offset):
                rank = flock.get_rank()
                assert rank != 0, "Rank 0 coordinates; workers must call compute."
                calls.append((rank, job))
                return job * job + offset

            expected = [[job * job + offset for job in jobs]] + [None] * (world - 1)
            trace = _run(fn, [list(jobs)] + [None] * (world - 1), expected, seed, compute)
            assert Counter(job for _, job in calls) == Counter(jobs), "Compute every job exactly once."
            links = {(0, rank) for rank in range(1, world)} | {(rank, 0) for rank in range(1, world)}
            workers = {rank for rank, _ in calls}
            required = {(0, rank) for rank in workers} | {(rank, 0) for rank in workers}
            _p2p_rules(trace, links, required)
    print("All checks passed.")


def _serial_life(board, shape, steps):
    height, width = shape
    for _ in range(steps):
        next_board = []
        for row in range(height):
            for col in range(width):
                count = sum(
                    board[((row + dy) % height) * width + (col + dx) % width]
                    for dy in (-1, 0, 1)
                    for dx in (-1, 0, 1)
                    if (dy, dx) != (0, 0)
                )
                next_board.append(count == 3 or (board[row * width + col] and count == 2))
        board = next_board
    return board


def check_life(fn):
    send, isend = flock.send, flock.isend

    async def observed_send(dst, value, tag=0):
        assert type(value) is bool, "Each message must contain one bool cell."
        await send(dst, value, tag=tag)

    def observed_isend(dst, value, tag=0):
        assert type(value) is bool, "Each message must contain one bool cell."
        return isend(dst, value, tag=tag)

    async def checked(cell, shape, steps):
        result = await fn(cell, shape, steps)
        assert type(result) is bool, "Return your cell's state as a bool."
        return result

    rng = random.Random(0)
    for height, width, steps in ((3, 3, 0), (3, 3, 1), (3, 5, 3), (5, 4, 5)):
        world = height * width
        board = [rng.random() < 0.4 for _ in range(world)]
        expected = _serial_life(board, (height, width), steps)
        routes = {
            (rank, peer): steps
            for rank in range(world)
            for peer in _life_neighbors(rank, (height, width))
            if steps
        }
        for seed in (0, 1, 7):
            with patch.object(flock, "send", observed_send), patch.object(flock, "isend", observed_isend):
                trace = _run(checked, board, expected, seed, (height, width), steps)
            _communication(trace, routes, {"send", "recv", "isend", "irecv"})
    print("All checks passed.")
