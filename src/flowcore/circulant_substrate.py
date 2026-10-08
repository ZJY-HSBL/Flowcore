from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .circulant import (
    kernel_to_spectrum,
    parallel_circulant_kronecker_scan,
    sequential_circulant_kronecker_scan,
)


@dataclass(slots=True)
class CirculantOperatorSequence:
    spectrum: Tensor
    local: Tensor
    bias: Tensor
    kernel: Tensor


class CirculantKroneckerSubstrate(nn.Module):
    """Directional, translation-structured cross-module Flow substrate.

    Module routing is circulant.  A real routing kernel k[t] is converted to a
    spectrum with FFT; scan composition multiplies spectra elementwise.  The
    family is exactly closed and can express directional cyclic transport.
    """

    def __init__(
        self,
        state_dim: int,
        block_size: int,
        *,
        stability_scale: float = 0.90,
    ) -> None:
        super().__init__()
        if state_dim % block_size != 0:
            raise ValueError("state_dim must be divisible by block_size")
        self.state_dim = state_dim
        self.block_size = block_size
        self.num_blocks = state_dim // block_size
        self.stability_scale = stability_scale

        raw_local = torch.empty(block_size, block_size)
        nn.init.orthogonal_(raw_local)
        self.raw_local = nn.Parameter(raw_local)
        self.raw_local_strength = nn.Parameter(torch.tensor(0.0))
        self.raw_kernel_prior = nn.Parameter(torch.zeros(self.num_blocks))

    def base_local(self) -> Tensor:
        w = self.raw_local
        row_sum = w.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
        normalized = w / row_sum
        strength = self.stability_scale * torch.sigmoid(
            self.raw_local_strength + 2.0
        )
        return normalized * strength

    def routing_kernel(self, route: Tensor, hold: Tensor) -> Tensor:
        """Build a stable real circular-routing kernel.

        Route selects directional offsets.  The mean Hold/update-rate controls
        how far the operator moves away from the identity delta kernel.
        """

        prior = torch.softmax(self.raw_kernel_prior, dim=-1)
        scores = route * prior
        scores = scores + 1e-8 * prior
        routed = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        alpha = hold.mean(dim=-1)
        identity = torch.zeros_like(routed)
        identity[..., 0] = 1.0
        return (1.0 - alpha)[..., None] * identity + alpha[..., None] * routed

    def build_operators(
        self,
        injection: Tensor,
        route: Tensor,
        hold: Tensor,
    ) -> CirculantOperatorSequence:
        if injection.shape[-1] != self.state_dim:
            raise ValueError("unexpected state dimension")
        if route.shape != hold.shape or route.shape[-1] != self.num_blocks:
            raise ValueError("route and hold must be [B,T,num_blocks]")
        if injection.shape[:2] != route.shape[:2]:
            raise ValueError("injection and controls must share batch/time")

        batch, time = injection.shape[:2]
        kernel = self.routing_kernel(route, hold)
        spectrum = kernel_to_spectrum(kernel)

        alpha = hold.mean(dim=-1)
        eye = torch.eye(
            self.block_size, device=injection.device, dtype=injection.dtype
        )
        base_local = self.base_local().to(dtype=injection.dtype)
        local = (
            (1.0 - alpha)[..., None, None] * eye
            + alpha[..., None, None] * base_local
        )

        bias = injection.view(batch, time, self.num_blocks, self.block_size)
        bias = alpha[..., None, None] * bias
        return CirculantOperatorSequence(spectrum, local, bias, kernel)

    def forward(
        self,
        injection: Tensor,
        route: Tensor,
        hold: Tensor,
        *,
        h0: Tensor | None = None,
        mode: str = "parallel",
    ) -> Tensor:
        ops = self.build_operators(injection, route, hold)
        batch = injection.shape[0]
        if h0 is None:
            h0_matrix = injection.new_zeros(
                batch, self.num_blocks, self.block_size
            )
        else:
            h0_matrix = h0.view(
                batch, self.num_blocks, self.block_size
            )

        if mode == "parallel":
            states = parallel_circulant_kronecker_scan(
                ops.spectrum, ops.local, ops.bias, h0_matrix
            )
        elif mode == "sequential":
            states = sequential_circulant_kronecker_scan(
                ops.spectrum, ops.local, ops.bias, h0_matrix
            )
        else:
            raise ValueError("mode must be 'parallel' or 'sequential'")

        return states.reshape(batch, injection.shape[1], self.state_dim)
