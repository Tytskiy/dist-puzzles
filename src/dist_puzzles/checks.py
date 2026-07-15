from typing import Any

import torch


def check(actual: list[Any], expected: list[Any], *, rtol: float = 0.0, atol: float = 0.0) -> None:
    if not isinstance(actual, list):
        raise AssertionError(
            f"check expected a list of per-rank results, got {type(actual).__name__}: {actual!r}"
        )

    if not isinstance(expected, list):
        raise AssertionError(
            f"check expected a list of per-rank expected values, got {type(expected).__name__}"
        )

    if len(actual) != len(expected):
        raise AssertionError(f"rank count mismatch: expected {len(expected)}, got {len(actual)}")

    for rank, (got, want) in enumerate(zip(actual, expected, strict=True)):
        _check(got, want, path=f"rank {rank}", rtol=rtol, atol=atol)


def _check(
    actual: Any,
    expected: Any,
    *,
    path: str,
    rtol: float,
    atol: float,
) -> None:
    if isinstance(expected, torch.Tensor):
        _check_tensor(actual, expected, path, rtol=rtol, atol=atol)
        return

    if isinstance(expected, list | tuple):
        _check_sequence(actual, expected, path=path, rtol=rtol, atol=atol)
        return

    if actual != expected:
        raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}")


def _check_tensor(actual: Any, expected: torch.Tensor, path: str, *, rtol: float, atol: float) -> None:
    if not isinstance(actual, torch.Tensor):
        raise AssertionError(f"{path}: expected torch.Tensor, got {type(actual).__name__}: {actual!r}")

    matches = torch.equal(actual, expected) if rtol == 0.0 and atol == 0.0 else torch.allclose(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
    )
    if not matches:
        raise AssertionError(
            f"{path}: expected {expected}, got {actual}"
        )


def _check_sequence(
    actual: Any,
    expected: list[Any] | tuple[Any, ...],
    *,
    path: str,
    rtol: float,
    atol: float,
) -> None:
    if not isinstance(actual, type(expected)):
        raise AssertionError(
            f"{path}: expected {type(expected).__name__}, got {type(actual).__name__}: {actual!r}"
        )

    if len(actual) != len(expected):
        raise AssertionError(f"{path}: length mismatch: expected {len(expected)}, got {len(actual)}")

    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        _check(got, want, path=f"{path}[{index}]", rtol=rtol, atol=atol)
