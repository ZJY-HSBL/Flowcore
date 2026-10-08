from __future__ import annotations

import torch
from torch import Tensor


def kernel_to_spectrum(kernel: Tensor) -> Tensor:
    """Convert a real circular-routing kernel to its complex spectrum."""

    return torch.fft.fft(kernel, dim=-1)


def apply_circulant_kronecker_linear(
    spectrum: Tensor,
    local: Tensor,
    state: Tensor,
) -> Tensor:
    """Apply circulant module routing and a shared local transform.

    The module axis is the second-to-last dimension.  Multiplication by a
    circulant routing matrix is implemented as FFT -> spectral multiply -> IFFT.
    """

    if state.shape[-2] != spectrum.shape[-1]:
        raise ValueError("state module dimension must match spectrum")
    if local.shape[-1] != local.shape[-2] or state.shape[-1] != local.shape[-1]:
        raise ValueError("local operator dimension must match state local dimension")

    state_frequency = torch.fft.fft(state, dim=-2)
    routed_frequency = state_frequency * spectrum.unsqueeze(-1)
    routed = torch.fft.ifft(routed_frequency, dim=-2).real
    return torch.matmul(routed, local.transpose(-1, -2))


def apply_circulant_kronecker_affine(
    spectrum: Tensor,
    local: Tensor,
    bias: Tensor,
    state: Tensor,
) -> Tensor:
    return apply_circulant_kronecker_linear(spectrum, local, state) + bias


def compose_circulant_kronecker_affine(
    later_spectrum: Tensor,
    later_local: Tensor,
    later_bias: Tensor,
    earlier_spectrum: Tensor,
    earlier_local: Tensor,
    earlier_bias: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compose two circulant-Kronecker affine transitions exactly."""

    spectrum = later_spectrum * earlier_spectrum
    local = later_local @ earlier_local
    bias = (
        apply_circulant_kronecker_linear(
            later_spectrum, later_local, earlier_bias
        )
        + later_bias
    )
    return spectrum, local, bias


def sequential_circulant_kronecker_scan(
    spectrum: Tensor,
    local: Tensor,
    bias: Tensor,
    h0: Tensor,
) -> Tensor:
    if spectrum.shape[:2] != local.shape[:2] or spectrum.shape[:2] != bias.shape[:2]:
        raise ValueError("operator tensors must share batch/time dimensions")
    if spectrum.shape[1] == 0:
        raise ValueError("time dimension must be non-empty")

    states: list[Tensor] = []
    h = h0
    for t in range(spectrum.shape[1]):
        h = apply_circulant_kronecker_affine(
            spectrum[:, t], local[:, t], bias[:, t], h
        )
        states.append(h)
    return torch.stack(states, dim=1)


def _identity_like(
    spectrum: Tensor,
    local: Tensor,
    bias: Tensor,
    count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    batch = spectrum.shape[0]
    modules = spectrum.shape[-1]
    local_dim = local.shape[-1]
    spectrum_identity = torch.ones(
        batch, count, modules, device=spectrum.device, dtype=spectrum.dtype
    )
    local_identity = torch.eye(
        local_dim, device=local.device, dtype=local.dtype
    ).view(1, 1, local_dim, local_dim)
    local_identity = local_identity.expand(batch, count, local_dim, local_dim)
    bias_identity = torch.zeros(
        batch, count, modules, local_dim, device=bias.device, dtype=bias.dtype
    )
    return spectrum_identity, local_identity, bias_identity


def _work_efficient_prefix(
    spectrum: Tensor,
    local: Tensor,
    bias: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    time = spectrum.shape[1]
    if time == 0:
        raise ValueError("time dimension must be non-empty")
    padded_time = 1 << (time - 1).bit_length()

    if padded_time != time:
        ispec, ilocal, ibias = _identity_like(
            spectrum, local, bias, padded_time - time
        )
        work_s = torch.cat((spectrum, ispec), dim=1)
        work_l = torch.cat((local, ilocal), dim=1)
        work_b = torch.cat((bias, ibias), dim=1)
    else:
        work_s, work_l, work_b = spectrum, local, bias

    step = 2
    while step <= padded_time:
        half = step // 2
        right = torch.arange(step - 1, padded_time, step, device=spectrum.device)
        left = right - half
        merged_s, merged_l, merged_b = compose_circulant_kronecker_affine(
            work_s.index_select(1, right),
            work_l.index_select(1, right),
            work_b.index_select(1, right),
            work_s.index_select(1, left),
            work_l.index_select(1, left),
            work_b.index_select(1, left),
        )
        work_s = torch.index_copy(work_s, 1, right, merged_s)
        work_l = torch.index_copy(work_l, 1, right, merged_l)
        work_b = torch.index_copy(work_b, 1, right, merged_b)
        step <<= 1

    root = torch.tensor([padded_time - 1], device=spectrum.device)
    ispec, ilocal, ibias = _identity_like(work_s, work_l, work_b, 1)
    work_s = torch.index_copy(work_s, 1, root, ispec)
    work_l = torch.index_copy(work_l, 1, root, ilocal)
    work_b = torch.index_copy(work_b, 1, root, ibias)

    step = padded_time
    while step >= 2:
        half = step // 2
        right = torch.arange(step - 1, padded_time, step, device=spectrum.device)
        left = right - half

        left_s = work_s.index_select(1, left)
        left_l = work_l.index_select(1, left)
        left_b = work_b.index_select(1, left)
        parent_s = work_s.index_select(1, right)
        parent_l = work_l.index_select(1, right)
        parent_b = work_b.index_select(1, right)

        right_s, right_l, right_b = compose_circulant_kronecker_affine(
            left_s, left_l, left_b, parent_s, parent_l, parent_b
        )

        work_s = torch.index_copy(work_s, 1, left, parent_s)
        work_l = torch.index_copy(work_l, 1, left, parent_l)
        work_b = torch.index_copy(work_b, 1, left, parent_b)
        work_s = torch.index_copy(work_s, 1, right, right_s)
        work_l = torch.index_copy(work_l, 1, right, right_l)
        work_b = torch.index_copy(work_b, 1, right, right_b)
        step >>= 1

    exclusive_s = work_s[:, :time]
    exclusive_l = work_l[:, :time]
    exclusive_b = work_b[:, :time]
    return compose_circulant_kronecker_affine(
        spectrum,
        local,
        bias,
        exclusive_s,
        exclusive_l,
        exclusive_b,
    )


def parallel_circulant_kronecker_scan(
    spectrum: Tensor,
    local: Tensor,
    bias: Tensor,
    h0: Tensor,
) -> Tensor:
    """Exact work-efficient parallel scan for circulant cross-module routing."""

    prefix_s, prefix_l, prefix_b = _work_efficient_prefix(
        spectrum, local, bias
    )
    time = spectrum.shape[1]
    h0_time = h0.unsqueeze(1).expand((-1, time) + tuple(h0.shape[1:]))
    return apply_circulant_kronecker_affine(
        prefix_s, prefix_l, prefix_b, h0_time
    )
