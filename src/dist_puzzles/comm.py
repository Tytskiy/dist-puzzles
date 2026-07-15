from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from flock import Tracer, TraceSummary


def comm_report(tracer: Tracer) -> str:
    return str(tracer)


def _used_links(summary: TraceSummary) -> set[tuple[int, int]]:
    return {
        (src, dst)
        for src in range(summary.world_size)
        for dst in range(summary.world_size)
        if summary.bytes_matrix[src][dst]
    }


def _normalize_rank_values(
    values: Sequence[int] | Any,
    *,
    world_size: int,
    label: str,
) -> tuple[int, ...]:
    if hasattr(values, "values"):
        values = values.values
    if len(values) != world_size:
        raise ValueError(f"{label} must have one entry per rank ({world_size}), got {len(values)}")
    return tuple(int(value) for value in values)


def check_comm(
    tracer: Tracer,
    *,
    total_bytes: int | None = None,
    total_messages: int | None = None,
    bytes_sent: Sequence[int] | Any | None = None,
    bytes_received: Sequence[int] | Any | None = None,
    allowed_links: set[tuple[int, int]] | None = None,
    required_links: set[tuple[int, int]] | None = None,
    forbidden_links: set[tuple[int, int]] | None = None,
    forbidden_collectives: set[str] | frozenset[str] | None = None,
) -> None:
    summary = tracer.summary()
    errors: list[str] = []

    if total_bytes is not None and summary.total_bytes != total_bytes:
        errors.append(f"total_bytes: expected {total_bytes}, got {summary.total_bytes}")

    if total_messages is not None and summary.total_messages != total_messages:
        errors.append(
            f"total_messages: expected {total_messages}, got {summary.total_messages}"
        )

    if bytes_sent is not None:
        expected = _normalize_rank_values(
            bytes_sent,
            world_size=summary.world_size,
            label="bytes_sent",
        )
        if summary.bytes_sent != expected:
            errors.append(f"bytes_sent: expected {list(expected)}, got {list(summary.bytes_sent)}")

    if bytes_received is not None:
        expected = _normalize_rank_values(
            bytes_received,
            world_size=summary.world_size,
            label="bytes_received",
        )
        if summary.bytes_received != expected:
            errors.append(
                f"bytes_received: expected {list(expected)}, got {list(summary.bytes_received)}"
            )

    used = _used_links(summary)

    if allowed_links is not None:
        disallowed = sorted(used - allowed_links)
        if disallowed:
            errors.append(f"disallowed links used: {disallowed}")

    if required_links is not None:
        missing = sorted(required_links - used)
        if missing:
            errors.append(f"required links missing: {missing}")

    if forbidden_links is not None:
        forbidden = sorted(used & forbidden_links)
        if forbidden:
            errors.append(f"forbidden links used: {forbidden}")

    if forbidden_collectives:
        used_collectives = {kind for kind, _ in summary.collective_counts}
        forbidden = sorted(used_collectives & set(forbidden_collectives))
        if forbidden:
            errors.append(f"forbidden collectives used: {forbidden}")

    if errors:
        report = comm_report(tracer)
        details = "\n".join(f"  - {line}" for line in errors)
        raise AssertionError(f"communication check failed:\n{details}\n\n{report}")
