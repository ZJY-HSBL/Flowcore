from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True)
class StateDiagnostics:
    """Numerical diagnostics for a state trajectory."""

    max_abs: float
    rms: float
    final_rms: float
    finite_fraction: float
    growth_ratio: float


def state_diagnostics(states: Tensor, *, eps: float = 1e-12) -> StateDiagnostics:
    """Summarize magnitude, finiteness, and temporal growth of a state sequence.

    ``states`` is expected to have batch/time as its first two dimensions.  The
    growth ratio compares RMS magnitude at the final time step with the first.
    It is a diagnostic only; it is not a stability proof.
    """

    if states.ndim < 3:
        raise ValueError("states must have [batch, time, ...state] dimensions")
    if states.shape[1] == 0:
        raise ValueError("time dimension must be non-empty")
    detached = states.detach().float()
    finite = torch.isfinite(detached)
    safe = torch.where(finite, detached, torch.zeros_like(detached))
    rms = safe.square().mean().sqrt()
    first = safe[:, 0].square().mean().sqrt()
    final = safe[:, -1].square().mean().sqrt()
    growth = final / first.clamp_min(eps)
    return StateDiagnostics(
        max_abs=float(safe.abs().amax().cpu()),
        rms=float(rms.cpu()),
        final_rms=float(final.cpu()),
        finite_fraction=float(finite.float().mean().cpu()),
        growth_ratio=float(growth.cpu()),
    )


def relative_state_error(reference: Tensor, candidate: Tensor, *, eps: float = 1e-12) -> float:
    """Maximum absolute error normalized by the reference maximum magnitude."""

    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have identical shapes")
    ref = reference.detach().float()
    cand = candidate.detach().float()
    scale = ref.abs().amax().clamp_min(eps)
    return float(((cand - ref).abs().amax() / scale).cpu())
