from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True)
class TransportBatch:
    injection: Tensor
    destination: Tensor
    payload: Tensor


@dataclass(slots=True)
class TransportMetrics:
    target_mse: float
    leakage_mse: float
    delivery_fraction: float
    source_fraction: float
    max_parallel_error: float | None = None


def make_transport_batch(
    batch_size: int,
    time_steps: int,
    num_modules: int,
    local_dim: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> TransportBatch:
    """Create a task that requires physical cross-module transport.

    A random payload is injected only into physical module 0 at t=0.  Every
    sample chooses a destination in modules 1..M-1.  The benchmark succeeds only
    when the final state places the payload in that destination module.
    """

    if num_modules < 2:
        raise ValueError("num_modules must be at least 2")
    if time_steps < 2:
        raise ValueError("time_steps must be at least 2")
    if batch_size <= 0 or local_dim <= 0:
        raise ValueError("batch_size and local_dim must be positive")

    device = torch.device(device)
    destination = torch.randint(
        1,
        num_modules,
        (batch_size,),
        device=device,
        generator=generator,
    )
    payload = torch.randn(
        batch_size,
        local_dim,
        device=device,
        generator=generator,
    )
    injection = torch.zeros(
        batch_size,
        time_steps,
        num_modules * local_dim,
        device=device,
    )
    injection[:, 0, :local_dim] = payload
    return TransportBatch(injection, destination, payload)


def make_transport_controls(
    destination: Tensor,
    route_at_transfer: Tensor,
    time_steps: int,
    num_modules: int,
) -> tuple[Tensor, Tensor]:
    """Build controls for inject -> transfer -> persist.

    t=0 accepts the injected payload, t=1 performs the learned route, and all
    later steps use Hold=0 so the resulting state should persist unchanged.
    """

    if route_at_transfer.shape != (destination.shape[0], num_modules):
        raise ValueError("route_at_transfer must be [batch, num_modules]")
    if time_steps < 2:
        raise ValueError("time_steps must be at least 2")

    batch = destination.shape[0]
    route = torch.ones(
        batch,
        time_steps,
        num_modules,
        device=route_at_transfer.device,
        dtype=route_at_transfer.dtype,
    )
    hold = torch.zeros_like(route)
    hold[:, 0] = 1.0
    hold[:, 1] = 1.0
    route[:, 1] = route_at_transfer
    return route, hold


def destination_state(
    states: Tensor,
    destination: Tensor,
    num_modules: int,
    local_dim: int,
) -> Tensor:
    """Gather each sample's final destination-module state."""

    final = states[:, -1].reshape(states.shape[0], num_modules, local_dim)
    batch_index = torch.arange(states.shape[0], device=states.device)
    return final[batch_index, destination]


def transport_loss(
    states: Tensor,
    destination: Tensor,
    payload: Tensor,
    num_modules: int,
    local_dim: int,
    *,
    leakage_weight: float = 0.10,
) -> Tensor:
    """MSE at the destination plus a penalty for energy in non-target modules."""

    final = states[:, -1].reshape(states.shape[0], num_modules, local_dim)
    batch_index = torch.arange(states.shape[0], device=states.device)
    target_state = final[batch_index, destination]
    target_mse = (target_state - payload).square().mean()

    mask = torch.ones(
        states.shape[0],
        num_modules,
        1,
        device=states.device,
        dtype=states.dtype,
    )
    mask[batch_index, destination] = 0.0
    leakage_mse = (final.square() * mask).sum() / (
        states.shape[0] * max(num_modules - 1, 1) * local_dim
    )
    return target_mse + leakage_weight * leakage_mse


@torch.no_grad()
def transport_metrics(
    states: Tensor,
    destination: Tensor,
    payload: Tensor,
    num_modules: int,
    local_dim: int,
    *,
    parallel_reference: Tensor | None = None,
) -> TransportMetrics:
    """Compute physical delivery and leakage metrics from a state trajectory."""

    final = states[:, -1].reshape(states.shape[0], num_modules, local_dim)
    batch_index = torch.arange(states.shape[0], device=states.device)
    target_state = final[batch_index, destination]
    target_mse = (target_state - payload).square().mean()

    energy = final.square().sum(dim=-1)
    target_energy = energy[batch_index, destination]
    source_energy = energy[:, 0]
    total_energy = energy.sum(dim=-1).clamp_min(1e-12)
    delivery = (target_energy / total_energy).mean()
    source_fraction = (source_energy / total_energy).mean()

    mask = torch.ones_like(energy)
    mask[batch_index, destination] = 0.0
    leakage_mse = (
        (energy * mask).sum()
        / (states.shape[0] * max(num_modules - 1, 1) * local_dim)
    )

    error = None
    if parallel_reference is not None:
        if parallel_reference.shape != states.shape:
            raise ValueError("parallel_reference must match states")
        error = float((states - parallel_reference).abs().amax().cpu())

    return TransportMetrics(
        target_mse=float(target_mse.cpu()),
        leakage_mse=float(leakage_mse.cpu()),
        delivery_fraction=float(delivery.cpu()),
        source_fraction=float(source_fraction.cpu()),
        max_parallel_error=error,
    )


def parameter_count(module: torch.nn.Module, *, trainable_only: bool = False) -> int:
    """Count scalar parameters."""

    parameters = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in parameters if p.requires_grad)
    return sum(p.numel() for p in parameters)
