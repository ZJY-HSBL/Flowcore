from __future__ import annotations

import torch
from torch import Tensor


def orthogonal_mixing_basis(
    size: int,
    *,
    seed: int = 17,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create a deterministic dense orthogonal module-mixing basis."""

    if size <= 0:
        raise ValueError("size must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    q, r = torch.linalg.qr(
        torch.randn(size, size, generator=generator, dtype=torch.float64)
    )
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (q * signs).to(dtype=dtype)


def materialize_module_matrix(module_gain: Tensor, basis: Tensor) -> Tensor:
    """Materialize Q diag(g) Q^T for diagnostics/tests."""

    if module_gain.shape[-1] != basis.shape[0] or basis.shape[0] != basis.shape[1]:
        raise ValueError("module_gain and basis dimensions are incompatible")
    return torch.einsum("ij,...j,kj->...ik", basis, module_gain, basis)


def apply_spectral_kronecker_linear(
    module_gain: Tensor,
    local: Tensor,
    state: Tensor,
    basis: Tensor,
) -> Tensor:
    """Apply the separable cross-module linear operator.

    State is represented as [..., modules, local_dim] and the map is
    H -> (Q diag(module_gain) Q^T) H local^T.
    """

    if state.shape[-2] != module_gain.shape[-1]:
        raise ValueError("state module dimension must match module_gain")
    if local.shape[-1] != local.shape[-2] or state.shape[-1] != local.shape[-1]:
        raise ValueError("local operator dimension must match state local dimension")
    if basis.shape != (state.shape[-2], state.shape[-2]):
        raise ValueError("basis must be square with one row/column per module")

    modal = torch.einsum("ij,...jd->...id", basis.transpose(0, 1), state)
    modal = modal * module_gain.unsqueeze(-1)
    mixed = torch.einsum("ij,...jd->...id", basis, modal)
    return torch.matmul(mixed, local.transpose(-1, -2))


def apply_spectral_kronecker_affine(
    module_gain: Tensor,
    local: Tensor,
    bias: Tensor,
    state: Tensor,
    basis: Tensor,
) -> Tensor:
    """Apply a spectral-Kronecker affine operator."""

    return apply_spectral_kronecker_linear(module_gain, local, state, basis) + bias


def compose_spectral_kronecker_affine(
    later_gain: Tensor,
    later_local: Tensor,
    later_bias: Tensor,
    earlier_gain: Tensor,
    earlier_local: Tensor,
    earlier_bias: Tensor,
    basis: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compose two spectral-Kronecker affine operators exactly.

    Every module-routing factor shares the same orthogonal basis Q, so
    R(g2) R(g1) = R(g2 * g1).  The operator family is therefore closed.
    """

    gain = later_gain * earlier_gain
    local = later_local @ earlier_local
    bias = (
        apply_spectral_kronecker_linear(
            later_gain, later_local, earlier_bias, basis
        )
        + later_bias
    )
    return gain, local, bias


def sequential_spectral_kronecker_scan(
    module_gain: Tensor,
    local: Tensor,
    bias: Tensor,
    h0: Tensor,
    basis: Tensor,
) -> Tensor:
    """Reference recurrent implementation."""

    if module_gain.shape[:2] != local.shape[:2] or module_gain.shape[:2] != bias.shape[:2]:
        raise ValueError("operator tensors must share batch/time dimensions")
    if module_gain.shape[1] == 0:
        raise ValueError("time dimension must be non-empty")

    states: list[Tensor] = []
    h = h0
    for t in range(module_gain.shape[1]):
        h = apply_spectral_kronecker_affine(
            module_gain[:, t],
            local[:, t],
            bias[:, t],
            h,
            basis,
        )
        states.append(h)
    return torch.stack(states, dim=1)


def _identity_like(
    module_gain: Tensor,
    local: Tensor,
    bias: Tensor,
    count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    batch = module_gain.shape[0]
    modules = module_gain.shape[-1]
    local_dim = local.shape[-1]
    gain_identity = torch.ones(
        batch, count, modules, device=module_gain.device, dtype=module_gain.dtype
    )
    local_identity = torch.eye(
        local_dim, device=local.device, dtype=local.dtype
    ).view(1, 1, local_dim, local_dim)
    local_identity = local_identity.expand(batch, count, local_dim, local_dim)
    bias_identity = torch.zeros(
        batch, count, modules, local_dim, device=bias.device, dtype=bias.dtype
    )
    return gain_identity, local_identity, bias_identity


def _work_efficient_prefix(
    module_gain: Tensor,
    local: Tensor,
    bias: Tensor,
    basis: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Inclusive work-efficient Blelloch scan for the closed operator family."""

    time = module_gain.shape[1]
    if time == 0:
        raise ValueError("time dimension must be non-empty")
    padded_time = 1 << (time - 1).bit_length()

    if padded_time != time:
        ig, il, ib = _identity_like(module_gain, local, bias, padded_time - time)
        work_g = torch.cat((module_gain, ig), dim=1)
        work_l = torch.cat((local, il), dim=1)
        work_b = torch.cat((bias, ib), dim=1)
    else:
        work_g, work_l, work_b = module_gain, local, bias

    step = 2
    while step <= padded_time:
        half = step // 2
        right = torch.arange(step - 1, padded_time, step, device=module_gain.device)
        left = right - half
        merged_g, merged_l, merged_b = compose_spectral_kronecker_affine(
            work_g.index_select(1, right),
            work_l.index_select(1, right),
            work_b.index_select(1, right),
            work_g.index_select(1, left),
            work_l.index_select(1, left),
            work_b.index_select(1, left),
            basis,
        )
        work_g = torch.index_copy(work_g, 1, right, merged_g)
        work_l = torch.index_copy(work_l, 1, right, merged_l)
        work_b = torch.index_copy(work_b, 1, right, merged_b)
        step <<= 1

    root = torch.tensor([padded_time - 1], device=module_gain.device)
    ig, il, ib = _identity_like(work_g, work_l, work_b, 1)
    work_g = torch.index_copy(work_g, 1, root, ig)
    work_l = torch.index_copy(work_l, 1, root, il)
    work_b = torch.index_copy(work_b, 1, root, ib)

    step = padded_time
    while step >= 2:
        half = step // 2
        right = torch.arange(step - 1, padded_time, step, device=module_gain.device)
        left = right - half

        left_g = work_g.index_select(1, left)
        left_l = work_l.index_select(1, left)
        left_b = work_b.index_select(1, left)
        parent_g = work_g.index_select(1, right)
        parent_l = work_l.index_select(1, right)
        parent_b = work_b.index_select(1, right)

        right_g, right_l, right_b = compose_spectral_kronecker_affine(
            left_g, left_l, left_b, parent_g, parent_l, parent_b, basis
        )

        work_g = torch.index_copy(work_g, 1, left, parent_g)
        work_l = torch.index_copy(work_l, 1, left, parent_l)
        work_b = torch.index_copy(work_b, 1, left, parent_b)
        work_g = torch.index_copy(work_g, 1, right, right_g)
        work_l = torch.index_copy(work_l, 1, right, right_l)
        work_b = torch.index_copy(work_b, 1, right, right_b)
        step >>= 1

    exclusive_g = work_g[:, :time]
    exclusive_l = work_l[:, :time]
    exclusive_b = work_b[:, :time]

    return compose_spectral_kronecker_affine(
        module_gain,
        local,
        bias,
        exclusive_g,
        exclusive_l,
        exclusive_b,
        basis,
    )


def parallel_spectral_kronecker_scan(
    module_gain: Tensor,
    local: Tensor,
    bias: Tensor,
    h0: Tensor,
    basis: Tensor,
) -> Tensor:
    """Exact parallel scan for structured cross-module dynamics."""

    if module_gain.shape[:2] != local.shape[:2] or module_gain.shape[:2] != bias.shape[:2]:
        raise ValueError("operator tensors must share batch/time dimensions")

    prefix_g, prefix_l, prefix_b = _work_efficient_prefix(
        module_gain, local, bias, basis
    )
    time = module_gain.shape[1]
    h0_time = h0.unsqueeze(1).expand((-1, time) + tuple(h0.shape[1:]))
    return apply_spectral_kronecker_affine(
        prefix_g, prefix_l, prefix_b, h0_time, basis
    )
