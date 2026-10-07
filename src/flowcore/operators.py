from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

OperatorKind = Literal["dense", "block", "diagonal"]
ScanAlgorithm = Literal["work_efficient", "hillis_steele"]


def _matmul(a: Tensor, b: Tensor, kind: OperatorKind) -> Tensor:
    if kind == "diagonal":
        return a * b
    return a @ b


def _matvec(a: Tensor, x: Tensor, kind: OperatorKind) -> Tensor:
    if kind == "diagonal":
        return a * x
    return torch.matmul(a, x.unsqueeze(-1)).squeeze(-1)


def compose_affine(
    later_a: Tensor,
    later_b: Tensor,
    earlier_a: Tensor,
    earlier_b: Tensor,
    *,
    kind: OperatorKind = "dense",
) -> tuple[Tensor, Tensor]:
    """Compose two affine operators in chronological order.

    If::

        x1 = earlier_a @ x0 + earlier_b
        x2 = later_a   @ x1 + later_b

    this returns ``(a, b)`` such that ``x2 = a @ x0 + b``.
    """

    a = _matmul(later_a, earlier_a, kind)
    b = _matvec(later_a, earlier_b, kind) + later_b
    return a, b


def apply_affine(a: Tensor, b: Tensor, x: Tensor, *, kind: OperatorKind = "dense") -> Tensor:
    """Apply an affine operator to a state tensor."""

    return _matvec(a, x, kind) + b


def sequential_affine_scan(
    a: Tensor,
    b: Tensor,
    h0: Tensor,
    *,
    kind: OperatorKind = "dense",
) -> Tensor:
    """Reference left-to-right recurrence for affine dynamics."""

    if a.shape[:2] != b.shape[:2]:
        raise ValueError("a and b must share batch/time dimensions")
    if a.shape[1] == 0:
        raise ValueError("time dimension must be non-empty")
    states: list[Tensor] = []
    h = h0
    for t in range(a.shape[1]):
        h = apply_affine(a[:, t], b[:, t], h, kind=kind)
        states.append(h)
    return torch.stack(states, dim=1)


def _hillis_steele_affine_prefix(
    a: Tensor,
    b: Tensor,
    *,
    kind: OperatorKind,
) -> tuple[Tensor, Tensor]:
    """Inclusive Hillis-Steele affine prefix scan.

    Sequential depth is O(log T), but total operator-composition work is
    O(T log T).  It is kept as a simple reference parallel algorithm.
    """

    prefix_a = a
    prefix_b = b
    time = a.shape[1]
    offset = 1
    while offset < time:
        later_a = prefix_a[:, offset:]
        later_b = prefix_b[:, offset:]
        earlier_a = prefix_a[:, :-offset]
        earlier_b = prefix_b[:, :-offset]
        comp_a, comp_b = compose_affine(
            later_a,
            later_b,
            earlier_a,
            earlier_b,
            kind=kind,
        )
        prefix_a = torch.cat((prefix_a[:, :offset], comp_a), dim=1)
        prefix_b = torch.cat((prefix_b[:, :offset], comp_b), dim=1)
        offset <<= 1
    return prefix_a, prefix_b


def _identity_like(a: Tensor, b: Tensor, *, kind: OperatorKind, count: int) -> tuple[Tensor, Tensor]:
    """Return ``count`` affine identity operators matching batch/operator shape."""

    batch = a.shape[0]
    if kind == "diagonal":
        eye_tail = torch.ones_like(a[:, :1])
    else:
        dim = a.shape[-1]
        eye = torch.eye(dim, device=a.device, dtype=a.dtype)
        tail_shape = (1,) * (a.ndim - 4) + (dim, dim)
        eye = eye.view(*tail_shape)
        eye_tail = eye.expand((batch, 1) + tuple(a.shape[2:]))
    eye_tail = eye_tail.expand((batch, count) + tuple(a.shape[2:]))
    zero_b = torch.zeros((batch, count) + tuple(b.shape[2:]), device=b.device, dtype=b.dtype)
    return eye_tail, zero_b


def _work_efficient_affine_prefix(
    a: Tensor,
    b: Tensor,
    *,
    kind: OperatorKind,
) -> tuple[Tensor, Tensor]:
    """Inclusive Blelloch scan over affine operators.

    The scan pads the time axis to a power of two, performs an upsweep and
    downsweep, then converts the exclusive result to an inclusive prefix.
    Compared with Hillis-Steele, this uses O(T) operator compositions while
    retaining O(log T) dependency depth.

    The implementation is functional (``torch.index_copy``) so gradients flow
    through every tree level and the kernel remains compatible with
    ``torch.compile`` for static shapes.
    """

    time = a.shape[1]
    if time == 0:
        raise ValueError("time dimension must be non-empty")
    padded_time = 1 << (time - 1).bit_length()
    if padded_time != time:
        pad_a, pad_b = _identity_like(a, b, kind=kind, count=padded_time - time)
        work_a = torch.cat((a, pad_a), dim=1)
        work_b = torch.cat((b, pad_b), dim=1)
    else:
        work_a, work_b = a, b

    # Upsweep: each right endpoint becomes the aggregate for its subtree.
    step = 2
    while step <= padded_time:
        half = step // 2
        right = torch.arange(step - 1, padded_time, step, device=a.device)
        left = right - half
        earlier_a = work_a.index_select(1, left)
        earlier_b = work_b.index_select(1, left)
        later_a = work_a.index_select(1, right)
        later_b = work_b.index_select(1, right)
        merged_a, merged_b = compose_affine(
            later_a, later_b, earlier_a, earlier_b, kind=kind
        )
        work_a = torch.index_copy(work_a, 1, right, merged_a)
        work_b = torch.index_copy(work_b, 1, right, merged_b)
        step <<= 1

    # Root becomes the exclusive identity prefix.
    root = torch.tensor([padded_time - 1], device=a.device)
    root_a, root_b = _identity_like(work_a, work_b, kind=kind, count=1)
    work_a = torch.index_copy(work_a, 1, root, root_a)
    work_b = torch.index_copy(work_b, 1, root, root_b)

    # Downsweep.  For non-commutative composition, the right child's prefix is
    # parent_prefix followed by the aggregate of the left subtree.
    step = padded_time
    while step >= 2:
        half = step // 2
        right = torch.arange(step - 1, padded_time, step, device=a.device)
        left = right - half
        left_aggregate_a = work_a.index_select(1, left)
        left_aggregate_b = work_b.index_select(1, left)
        parent_prefix_a = work_a.index_select(1, right)
        parent_prefix_b = work_b.index_select(1, right)

        right_prefix_a, right_prefix_b = compose_affine(
            left_aggregate_a,
            left_aggregate_b,
            parent_prefix_a,
            parent_prefix_b,
            kind=kind,
        )
        work_a = torch.index_copy(work_a, 1, left, parent_prefix_a)
        work_b = torch.index_copy(work_b, 1, left, parent_prefix_b)
        work_a = torch.index_copy(work_a, 1, right, right_prefix_a)
        work_b = torch.index_copy(work_b, 1, right, right_prefix_b)
        step >>= 1

    exclusive_a = work_a[:, :time]
    exclusive_b = work_b[:, :time]
    inclusive_a, inclusive_b = compose_affine(
        a,
        b,
        exclusive_a,
        exclusive_b,
        kind=kind,
    )
    return inclusive_a, inclusive_b


def parallel_affine_scan(
    a: Tensor,
    b: Tensor,
    h0: Tensor,
    *,
    kind: OperatorKind = "dense",
    algorithm: ScanAlgorithm = "work_efficient",
) -> Tensor:
    """Parallel scan over affine state-transition operators.

    Parameters
    ----------
    algorithm:
        ``"work_efficient"`` uses a Blelloch tree with O(T) total operator
        compositions and O(log T) dependency depth. ``"hillis_steele"`` keeps
        the simpler O(T log T)-work implementation for comparison.
    """

    if a.ndim < 3 or b.ndim < 3:
        raise ValueError("a and b must include batch and time dimensions")
    if a.shape[:2] != b.shape[:2]:
        raise ValueError("a and b must share batch/time dimensions")
    if a.shape[1] == 0:
        raise ValueError("time dimension must be non-empty")

    if algorithm == "work_efficient":
        prefix_a, prefix_b = _work_efficient_affine_prefix(a, b, kind=kind)
    elif algorithm == "hillis_steele":
        prefix_a, prefix_b = _hillis_steele_affine_prefix(a, b, kind=kind)
    else:
        raise ValueError("algorithm must be 'work_efficient' or 'hillis_steele'")

    time = a.shape[1]
    h0_time = h0.unsqueeze(1).expand((-1, time) + tuple(h0.shape[1:]))
    return apply_affine(prefix_a, prefix_b, h0_time, kind=kind)


def make_compiled_scan(
    *,
    kind: OperatorKind = "block",
    algorithm: ScanAlgorithm = "work_efficient",
    dynamic: bool = False,
    mode: str = "reduce-overhead",
):
    """Create a ``torch.compile``-wrapped scan callable.

    Compilation is intentionally opt-in because compile latency can dominate
    short experiments.  Static sequence shapes are the primary target for v0.2.
    """

    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile requires PyTorch 2.x")

    def scan(a: Tensor, b: Tensor, h0: Tensor) -> Tensor:
        return parallel_affine_scan(a, b, h0, kind=kind, algorithm=algorithm)

    return torch.compile(scan, dynamic=dynamic, mode=mode)


def scan_composition_count(time: int, algorithm: ScanAlgorithm = "work_efficient") -> int:
    """Count affine compositions performed by the abstract scan algorithm.

    This is an algorithmic work metric, not a FLOP or wall-clock measurement.
    The work-efficient count includes the final exclusive-to-inclusive compose.
    """

    if time <= 0:
        raise ValueError("time must be positive")
    if algorithm == "hillis_steele":
        total = 0
        offset = 1
        while offset < time:
            total += time - offset
            offset <<= 1
        return total
    if algorithm == "work_efficient":
        padded = 1 << (time - 1).bit_length()
        return 2 * (padded - 1) + time
    raise ValueError("algorithm must be 'work_efficient' or 'hillis_steele'")
