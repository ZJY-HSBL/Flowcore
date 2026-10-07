from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

OperatorKind = Literal["dense", "block", "diagonal"]


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
    """Compose two affine operators.

    The operators are applied in chronological order::

        x1 = earlier_a @ x0 + earlier_b
        x2 = later_a   @ x1 + later_b

    and this function returns ``(a, b)`` such that ``x2 = a @ x0 + b``.

    For ``kind='block'`` the trailing dimensions are ``[..., n_blocks, d, d]``
    and the block axis is treated as an arbitrary batch dimension.  For
    ``kind='diagonal'`` the matrices are represented by their diagonals.
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
    """Reference left-to-right recurrence for affine dynamics.

    ``a`` and ``b`` use shape ``[batch, time, ...]``.  The returned tensor has
    shape ``[batch, time, ...state...]`` and contains the post-update state for
    every time step.
    """

    if a.shape[:2] != b.shape[:2]:
        raise ValueError("a and b must share batch/time dimensions")
    states: list[Tensor] = []
    h = h0
    for t in range(a.shape[1]):
        h = apply_affine(a[:, t], b[:, t], h, kind=kind)
        states.append(h)
    return torch.stack(states, dim=1)


def parallel_affine_scan(
    a: Tensor,
    b: Tensor,
    h0: Tensor,
    *,
    kind: OperatorKind = "dense",
) -> Tensor:
    """Inclusive Hillis-Steele scan over affine state-transition operators.

    The implementation has ``O(log T)`` sequential Python/PyTorch stages.  Every
    stage composes all eligible time positions in parallel.  It is exact (up to
    floating-point associativity) for affine operators whose representation is
    closed under composition.

    Notes
    -----
    * ``dense`` is mainly a correctness/reference backend.  Composition uses
      matrix-matrix products and is expensive for large state sizes.
    * ``block`` keeps independent small square blocks and is the default scalable
      research backend.
    * ``diagonal`` is the cheapest fully scan-friendly backend.
    """

    if a.ndim < 3 or b.ndim < 3:
        raise ValueError("a and b must include batch and time dimensions")
    if a.shape[:2] != b.shape[:2]:
        raise ValueError("a and b must share batch/time dimensions")

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

    # The prefix operator at t maps h0 -> h_{t+1}.
    h0_time = h0.unsqueeze(1).expand((-1, time) + tuple(h0.shape[1:]))
    return apply_affine(prefix_a, prefix_b, h0_time, kind=kind)
