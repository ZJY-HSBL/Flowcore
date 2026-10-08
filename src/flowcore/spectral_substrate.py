from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .kronecker import (
    materialize_module_matrix,
    orthogonal_mixing_basis,
    parallel_spectral_kronecker_scan,
    sequential_spectral_kronecker_scan,
)


@dataclass(slots=True)
class SpectralOperatorSequence:
    module_gain: Tensor
    local: Tensor
    bias: Tensor


class SpectralKroneckerSubstrate(nn.Module):
    """Exact cross-module Flow substrate using a closed Kronecker family.

    State is shaped [modules, local_dim].  Each transition is
    H[t+1] = R[t] H[t] L[t]^T + B[t],
    where R[t] = Q diag(g[t]) Q^T.

    v0.3 deliberately trades arbitrary directed routing for exact algebraic
    closure and parallel scan.
    """

    def __init__(
        self,
        state_dim: int,
        block_size: int,
        *,
        stability_scale: float = 0.90,
        mixing_basis_seed: int = 17,
    ) -> None:
        super().__init__()
        if state_dim % block_size != 0:
            raise ValueError("state_dim must be divisible by block_size")
        self.state_dim = state_dim
        self.block_size = block_size
        self.num_blocks = state_dim // block_size
        self.stability_scale = stability_scale

        basis = orthogonal_mixing_basis(self.num_blocks, seed=mixing_basis_seed)
        self.register_buffer("mixing_basis", basis)

        raw_local = torch.empty(block_size, block_size)
        nn.init.orthogonal_(raw_local)
        self.raw_local = nn.Parameter(raw_local)
        self.raw_local_strength = nn.Parameter(torch.tensor(0.0))
        self.raw_mode_strength = nn.Parameter(torch.zeros(self.num_blocks))

    def base_local(self) -> Tensor:
        w = self.raw_local
        row_sum = w.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
        normalized = w / row_sum
        strength = self.stability_scale * torch.sigmoid(self.raw_local_strength + 2.0)
        return normalized * strength

    def mode_strength(self) -> Tensor:
        return self.stability_scale * torch.sigmoid(self.raw_mode_strength + 2.0)

    def build_operators(
        self,
        injection: Tensor,
        route: Tensor,
        hold: Tensor,
    ) -> SpectralOperatorSequence:
        if injection.shape[-1] != self.state_dim:
            raise ValueError("unexpected state dimension")
        if route.shape != hold.shape or route.shape[-1] != self.num_blocks:
            raise ValueError("route and hold must be [B,T,num_blocks]")
        if injection.shape[:2] != route.shape[:2]:
            raise ValueError("injection and controls must share batch/time")

        batch, time = injection.shape[:2]

        # Route/Hold are modal coordinates in this backend.
        # hold=0 -> modal identity/persistence.
        # hold=1 -> routed long-term modal dynamics.
        target_gain = route * self.mode_strength()[None, None, :]
        module_gain = (1.0 - hold) + hold * target_gain

        # A shared local factor preserves a single Kronecker term under composition.
        local_alpha = hold.mean(dim=-1)
        eye = torch.eye(
            self.block_size, device=injection.device, dtype=injection.dtype
        )
        base_local = self.base_local().to(dtype=injection.dtype)
        local = (
            (1.0 - local_alpha)[..., None, None] * eye
            + local_alpha[..., None, None] * base_local
        )

        bias = injection.view(batch, time, self.num_blocks, self.block_size)
        bias = local_alpha[..., None, None] * bias
        return SpectralOperatorSequence(module_gain, local, bias)

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
            h0_matrix = injection.new_zeros(batch, self.num_blocks, self.block_size)
        else:
            h0_matrix = h0.view(batch, self.num_blocks, self.block_size)

        basis = self.mixing_basis.to(device=injection.device, dtype=injection.dtype)
        if mode == "parallel":
            states = parallel_spectral_kronecker_scan(
                ops.module_gain, ops.local, ops.bias, h0_matrix, basis
            )
        elif mode == "sequential":
            states = sequential_spectral_kronecker_scan(
                ops.module_gain, ops.local, ops.bias, h0_matrix, basis
            )
        else:
            raise ValueError("mode must be 'parallel' or 'sequential'")

        return states.reshape(batch, injection.shape[1], self.state_dim)

    def route_matrix(self, module_gain: Tensor) -> Tensor:
        basis = self.mixing_basis.to(
            device=module_gain.device, dtype=module_gain.dtype
        )
        return materialize_module_matrix(module_gain, basis)

    def modal_stability_bound(self) -> Tensor:
        return self.mode_strength().abs().amax()
