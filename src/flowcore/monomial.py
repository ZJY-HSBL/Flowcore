from __future__ import annotations

import torch
from torch import Tensor


def _gather_modules(state: Tensor, permutation: Tensor) -> Tensor:
    if state.shape[-2] != permutation.shape[-1]:
        raise ValueError("state module dimension must match permutation")
    index = permutation.unsqueeze(-1).expand(
        permutation.shape + (state.shape[-1],)
    )
    return torch.gather(state, -2, index)


def apply_monomial_linear(
    permutation: Tensor,
    gain: Tensor,
    local: Tensor,
    state: Tensor,
) -> Tensor:
    """Apply an arbitrary one-to-one module permutation with per-output gain.

    Convention:
        output[i] = gain[i] * state[permutation[i]] @ local.T
    """

    if permutation.dtype != torch.long:
        raise ValueError("permutation must use torch.long")
    if permutation.shape != gain.shape:
        raise ValueError("permutation and gain must have identical shape")
    if local.shape[-1] != local.shape[-2] or state.shape[-1] != local.shape[-1]:
        raise ValueError("local operator dimension must match state local dimension")
    routed = _gather_modules(state, permutation)
    routed = routed * gain.unsqueeze(-1)
    return torch.matmul(routed, local.transpose(-1, -2))


def apply_monomial_affine(
    permutation: Tensor,
    gain: Tensor,
    local: Tensor,
    bias: Tensor,
    state: Tensor,
) -> Tensor:
    return apply_monomial_linear(permutation, gain, local, state) + bias


def compose_monomial_affine(
    later_permutation: Tensor,
    later_gain: Tensor,
    later_local: Tensor,
    later_bias: Tensor,
    earlier_permutation: Tensor,
    earlier_gain: Tensor,
    earlier_local: Tensor,
    earlier_bias: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compose two monomial-Kronecker affine operators exactly."""

    if later_permutation.dtype != torch.long or earlier_permutation.dtype != torch.long:
        raise ValueError("permutations must use torch.long")

    # z[i] reads y[later_perm[i]], while y[j] reads x[earlier_perm[j]].
    permutation = torch.gather(
        earlier_permutation, -1, later_permutation
    )
    earlier_gain_reindexed = torch.gather(
        earlier_gain, -1, later_permutation
    )
    gain = later_gain * earlier_gain_reindexed
    local = later_local @ earlier_local
    bias = (
        apply_monomial_linear(
            later_permutation,
            later_gain,
            later_local,
            earlier_bias,
        )
        + later_bias
    )
    return permutation, gain, local, bias


def sequential_monomial_scan(
    permutation: Tensor,
    gain: Tensor,
    local: Tensor,
    bias: Tensor,
    h0: Tensor,
) -> Tensor:
    if permutation.shape[:2] != gain.shape[:2]:
        raise ValueError("operator tensors must share batch/time")
    if permutation.shape[:2] != local.shape[:2] or permutation.shape[:2] != bias.shape[:2]:
        raise ValueError("operator tensors must share batch/time")
    if permutation.shape[1] == 0:
        raise ValueError("time dimension must be non-empty")

    states: list[Tensor] = []
    h = h0
    for t in range(permutation.shape[1]):
        h = apply_monomial_affine(
            permutation[:, t],
            gain[:, t],
            local[:, t],
            bias[:, t],
            h,
        )
        states.append(h)
    return torch.stack(states, dim=1)


def _identity_like(
    permutation: Tensor,
    gain: Tensor,
    local: Tensor,
    bias: Tensor,
    count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    batch = permutation.shape[0]
    modules = permutation.shape[-1]
    local_dim = local.shape[-1]
    identity_perm = torch.arange(
        modules, device=permutation.device, dtype=torch.long
    ).view(1, 1, modules)
    identity_perm = identity_perm.expand(batch, count, modules)
    identity_gain = torch.ones(
        batch,
        count,
        modules,
        device=gain.device,
        dtype=gain.dtype,
    )
    identity_local = torch.eye(
        local_dim, device=local.device, dtype=local.dtype
    ).view(1, 1, local_dim, local_dim)
    identity_local = identity_local.expand(
        batch, count, local_dim, local_dim
    )
    identity_bias = torch.zeros(
        batch,
        count,
        modules,
        local_dim,
        device=bias.device,
        dtype=bias.dtype,
    )
    return identity_perm, identity_gain, identity_local, identity_bias


def _work_efficient_prefix(
    permutation: Tensor,
    gain: Tensor,
    local: Tensor,
    bias: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    time = permutation.shape[1]
    if time == 0:
        raise ValueError("time dimension must be non-empty")
    padded_time = 1 << (time - 1).bit_length()

    if padded_time != time:
        ip, ig, il, ib = _identity_like(
            permutation, gain, local, bias, padded_time - time
        )
        work_p = torch.cat((permutation, ip), dim=1)
        work_g = torch.cat((gain, ig), dim=1)
        work_l = torch.cat((local, il), dim=1)
        work_b = torch.cat((bias, ib), dim=1)
    else:
        work_p, work_g, work_l, work_b = (
            permutation,
            gain,
            local,
            bias,
        )

    step = 2
    while step <= padded_time:
        half = step // 2
        right = torch.arange(
            step - 1, padded_time, step, device=permutation.device
        )
        left = right - half
        merged = compose_monomial_affine(
            work_p.index_select(1, right),
            work_g.index_select(1, right),
            work_l.index_select(1, right),
            work_b.index_select(1, right),
            work_p.index_select(1, left),
            work_g.index_select(1, left),
            work_l.index_select(1, left),
            work_b.index_select(1, left),
        )
        work_p = torch.index_copy(work_p, 1, right, merged[0])
        work_g = torch.index_copy(work_g, 1, right, merged[1])
        work_l = torch.index_copy(work_l, 1, right, merged[2])
        work_b = torch.index_copy(work_b, 1, right, merged[3])
        step <<= 1

    root = torch.tensor(
        [padded_time - 1], device=permutation.device
    )
    ip, ig, il, ib = _identity_like(
        work_p, work_g, work_l, work_b, 1
    )
    work_p = torch.index_copy(work_p, 1, root, ip)
    work_g = torch.index_copy(work_g, 1, root, ig)
    work_l = torch.index_copy(work_l, 1, root, il)
    work_b = torch.index_copy(work_b, 1, root, ib)

    step = padded_time
    while step >= 2:
        half = step // 2
        right = torch.arange(
            step - 1, padded_time, step, device=permutation.device
        )
        left = right - half

        left_p = work_p.index_select(1, left)
        left_g = work_g.index_select(1, left)
        left_l = work_l.index_select(1, left)
        left_b = work_b.index_select(1, left)
        parent_p = work_p.index_select(1, right)
        parent_g = work_g.index_select(1, right)
        parent_l = work_l.index_select(1, right)
        parent_b = work_b.index_select(1, right)

        right_values = compose_monomial_affine(
            left_p,
            left_g,
            left_l,
            left_b,
            parent_p,
            parent_g,
            parent_l,
            parent_b,
        )

        work_p = torch.index_copy(work_p, 1, left, parent_p)
        work_g = torch.index_copy(work_g, 1, left, parent_g)
        work_l = torch.index_copy(work_l, 1, left, parent_l)
        work_b = torch.index_copy(work_b, 1, left, parent_b)
        work_p = torch.index_copy(work_p, 1, right, right_values[0])
        work_g = torch.index_copy(work_g, 1, right, right_values[1])
        work_l = torch.index_copy(work_l, 1, right, right_values[2])
        work_b = torch.index_copy(work_b, 1, right, right_values[3])
        step >>= 1

    exclusive_p = work_p[:, :time]
    exclusive_g = work_g[:, :time]
    exclusive_l = work_l[:, :time]
    exclusive_b = work_b[:, :time]
    return compose_monomial_affine(
        permutation,
        gain,
        local,
        bias,
        exclusive_p,
        exclusive_g,
        exclusive_l,
        exclusive_b,
    )


def parallel_monomial_scan(
    permutation: Tensor,
    gain: Tensor,
    local: Tensor,
    bias: Tensor,
    h0: Tensor,
) -> Tensor:
    """Exact work-efficient scan for arbitrary one-to-one module routing."""

    prefix_p, prefix_g, prefix_l, prefix_b = _work_efficient_prefix(
        permutation, gain, local, bias
    )
    time = permutation.shape[1]
    h0_time = h0.unsqueeze(1).expand(
        (-1, time) + tuple(h0.shape[1:])
    )
    return apply_monomial_affine(
        prefix_p,
        prefix_g,
        prefix_l,
        prefix_b,
        h0_time,
    )
