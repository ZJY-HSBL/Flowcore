from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True)
class PermutationBatch:
    injection: Tensor
    task_id: Tensor
    payload: Tensor
    target: Tensor


@dataclass(slots=True)
class PermutationMetrics:
    mse: float
    relative_mse: float
    slot_accuracy: float
    max_parallel_error: float | None = None


def is_cyclic_permutation(permutation: Tensor) -> bool:
    """Return True when a gather permutation is a single cyclic shift."""

    if permutation.ndim != 1:
        raise ValueError("permutation must be one-dimensional")
    modules = permutation.numel()
    base = torch.arange(modules, device=permutation.device)
    offsets = torch.remainder(permutation - base, modules)
    return bool(torch.all(offsets == offsets[0]).item())


def cyclic_permutation_bank(num_tasks: int, num_modules: int) -> Tensor:
    """Create distinct non-identity cyclic gather permutations."""

    if num_modules < 2:
        raise ValueError("num_modules must be at least 2")
    if num_tasks <= 0 or num_tasks > num_modules - 1:
        raise ValueError("num_tasks must be in [1, num_modules-1]")
    base = torch.arange(num_modules)
    rows = []
    for shift in range(1, num_tasks + 1):
        rows.append(torch.remainder(base - shift, num_modules))
    return torch.stack(rows)


def arbitrary_permutation_bank(
    num_tasks: int,
    num_modules: int,
    *,
    seed: int = 123,
) -> Tensor:
    """Create unique permutations that are not cyclic shifts or identity."""

    if num_modules < 3:
        raise ValueError("num_modules must be at least 3")
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    rows: list[Tensor] = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    max_attempts = 10000
    while len(rows) < num_tasks and attempts < max_attempts:
        attempts += 1
        candidate = torch.randperm(num_modules, generator=generator)
        key = tuple(int(x) for x in candidate.tolist())
        if key in seen or is_cyclic_permutation(candidate):
            continue
        seen.add(key)
        rows.append(candidate)
    if len(rows) != num_tasks:
        raise ValueError(
            "could not generate requested number of unique non-cyclic permutations"
        )
    return torch.stack(rows)


def permutation_matrix(permutation: Tensor, *, dtype: torch.dtype = torch.float32) -> Tensor:
    """Materialize P with y[i] = x[permutation[i]]."""

    if permutation.ndim != 1:
        raise ValueError("permutation must be one-dimensional")
    modules = permutation.numel()
    matrix = torch.zeros(
        modules,
        modules,
        dtype=dtype,
        device=permutation.device,
    )
    matrix[torch.arange(modules, device=permutation.device), permutation] = 1.0
    return matrix


def _relative_frobenius_error(target: Tensor, approximation: Tensor) -> float:
    numerator = (target - approximation).square().sum().sqrt()
    denominator = target.square().sum().sqrt().clamp_min(1e-12)
    return float((numerator / denominator).cpu())


def block_projection_error(permutation: Tensor) -> float:
    """Best diagonal-module approximation error for a target permutation."""

    target = permutation_matrix(permutation)
    approximation = torch.diag(torch.diagonal(target))
    return _relative_frobenius_error(target, approximation)


def circulant_projection_error(permutation: Tensor) -> float:
    """Best unconstrained circulant approximation in normalized Frobenius norm."""

    target = permutation_matrix(permutation)
    modules = permutation.numel()
    identity = torch.eye(modules, device=target.device, dtype=target.dtype)
    approximation = torch.zeros_like(target)
    # Cyclic shift permutation matrices are orthogonal under Frobenius product.
    for shift in range(modules):
        basis = torch.roll(identity, shifts=shift, dims=0)
        coefficient = (target * basis).sum() / modules
        approximation = approximation + coefficient * basis
    return _relative_frobenius_error(target, approximation)


def spectral_projection_error(
    permutation: Tensor,
    basis: Tensor,
    *,
    positive_unit_gain: bool = True,
) -> float:
    """Best shared-basis Q diag(g) Q^T approximation error.

    With positive_unit_gain=True, coefficients are clamped to [0,1] to match the
    current FlowCore spectral substrate's effective modal-gain restriction.
    """

    target = permutation_matrix(
        permutation,
        dtype=basis.dtype,
    ).to(basis.device)
    modal = basis.transpose(0, 1) @ target @ basis
    gain = torch.diagonal(modal)
    if positive_unit_gain:
        gain = gain.clamp(0.0, 1.0)
    approximation = basis @ torch.diag(gain) @ basis.transpose(0, 1)
    return _relative_frobenius_error(target, approximation)


def make_permutation_batch(
    batch_size: int,
    time_steps: int,
    permutations: Tensor,
    local_dim: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> PermutationBatch:
    """Create a simultaneous multi-source permutation-routing batch."""

    if permutations.ndim != 2:
        raise ValueError("permutations must have shape [tasks, modules]")
    if batch_size <= 0 or local_dim <= 0:
        raise ValueError("batch_size and local_dim must be positive")
    if time_steps < 2:
        raise ValueError("time_steps must be at least 2")

    device = torch.device(device)
    permutations = permutations.to(device)
    tasks, modules = permutations.shape
    task_id = torch.randint(
        0,
        tasks,
        (batch_size,),
        device=device,
        generator=generator,
    )
    payload = torch.randn(
        batch_size,
        modules,
        local_dim,
        device=device,
        generator=generator,
    )
    selected = permutations[task_id]
    gather_index = selected.unsqueeze(-1).expand(-1, -1, local_dim)
    target = torch.gather(payload, 1, gather_index)

    injection = torch.zeros(
        batch_size,
        time_steps,
        modules * local_dim,
        device=device,
    )
    injection[:, 0] = payload.reshape(batch_size, modules * local_dim)
    return PermutationBatch(injection, task_id, payload, target)


def make_permutation_controls(
    route_at_transfer: Tensor,
    time_steps: int,
) -> tuple[Tensor, Tensor]:
    """Accept all payloads, apply one global route, then persist."""

    if route_at_transfer.ndim != 2:
        raise ValueError("route_at_transfer must be [batch, modules]")
    if time_steps < 2:
        raise ValueError("time_steps must be at least 2")

    batch, modules = route_at_transfer.shape
    route = torch.ones(
        batch,
        time_steps,
        modules,
        device=route_at_transfer.device,
        dtype=route_at_transfer.dtype,
    )
    hold = torch.zeros_like(route)
    hold[:, 0] = 1.0
    hold[:, 1] = 1.0
    route[:, 1] = route_at_transfer
    return route, hold


def permutation_loss(
    states: Tensor,
    target: Tensor,
    num_modules: int,
    local_dim: int,
) -> Tensor:
    final = states[:, -1].reshape(states.shape[0], num_modules, local_dim)
    return (final - target).square().mean()


@torch.no_grad()
def permutation_metrics(
    states: Tensor,
    payload: Tensor,
    target: Tensor,
    task_id: Tensor,
    permutations: Tensor,
    *,
    parallel_reference: Tensor | None = None,
) -> PermutationMetrics:
    """Measure value error and source-slot recovery accuracy."""

    batch, modules, local_dim = payload.shape
    final = states[:, -1].reshape(batch, modules, local_dim)
    mse = (final - target).square().mean()
    baseline = target.square().mean().clamp_min(1e-12)

    # Identify which input payload each output slot most closely resembles.
    pairwise = (
        final[:, :, None, :] - payload[:, None, :, :]
    ).square().mean(dim=-1)
    predicted_source = pairwise.argmin(dim=-1)
    expected_source = permutations.to(task_id.device)[task_id]
    slot_accuracy = (predicted_source == expected_source).float().mean()

    error = None
    if parallel_reference is not None:
        error = float(
            (states - parallel_reference).abs().amax().detach().cpu()
        )

    return PermutationMetrics(
        mse=float(mse.cpu()),
        relative_mse=float((mse / baseline).cpu()),
        slot_accuracy=float(slot_accuracy.cpu()),
        max_parallel_error=error,
    )
