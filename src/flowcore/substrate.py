from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .operators import parallel_affine_scan, sequential_affine_scan


@dataclass(slots=True)
class OperatorSequence:
    a: Tensor
    b: Tensor
    kind: str


class ParallelFlowSubstrate(nn.Module):
    """Block-diagonal dynamic substrate with exact affine parallel scan.

    Each local block acts as a small microcircuit.  The long-term matrix ``W`` is
    trainable.  The controller supplies a per-time/per-block route gain and hold
    (update-rate) signal.  Because no cross-block coupling is introduced inside a
    scan pass, block-affine operators remain closed under composition.

    Rich cross-module communication is intended to happen through hierarchical
    macro-steps/refinement passes in later versions rather than by destroying the
    algebra that makes exact parallel scan possible.
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

        raw = torch.empty(self.num_blocks, block_size, block_size)
        nn.init.orthogonal_(raw.view(-1, block_size))
        # A bounded scalar controls each block's long-term dynamic strength.
        self.raw_blocks = nn.Parameter(raw)
        self.raw_strength = nn.Parameter(torch.zeros(self.num_blocks))

    def base_blocks(self) -> Tensor:
        # Normalize row absolute sums to <= 1 then apply learned bounded strength.
        w = self.raw_blocks
        row_sum = w.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
        normalized = w / row_sum
        strength = self.stability_scale * torch.sigmoid(self.raw_strength + 2.0)
        return normalized * strength[:, None, None]

    def build_operators(self, injection: Tensor, route: Tensor, hold: Tensor) -> OperatorSequence:
        """Build ``h[t+1] = A[t] h[t] + b[t]`` in block representation.

        Parameters
        ----------
        injection:
            ``[B,T,state_dim]`` candidate input drive.
        route:
            ``[B,T,num_blocks]`` route gain for each local module.
        hold:
            ``[B,T,num_blocks]`` update rate alpha.  Small alpha means state is
            retained; large alpha means the module rapidly accepts new dynamics.
        """

        if injection.shape[-1] != self.state_dim:
            raise ValueError("unexpected state dimension")
        if route.shape != hold.shape or route.shape[-1] != self.num_blocks:
            raise ValueError("route and hold must be [B,T,num_blocks]")
        if injection.shape[:2] != route.shape[:2]:
            raise ValueError("injection and controls must share batch/time")

        bsz, time = injection.shape[:2]
        w = self.base_blocks()[None, None]  # [1,1,K,D,D]
        alpha = hold[..., :, None, None]
        gate = route[..., :, None, None]
        eye = torch.eye(self.block_size, device=injection.device, dtype=injection.dtype)
        eye = eye.view(1, 1, 1, self.block_size, self.block_size)
        a = (1.0 - alpha) * eye + alpha * gate * w

        b = injection.view(bsz, time, self.num_blocks, self.block_size)
        b = hold[..., :, None] * b
        return OperatorSequence(a=a, b=b, kind="block")

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
        bsz = injection.shape[0]
        if h0 is None:
            h0 = injection.new_zeros(bsz, self.num_blocks, self.block_size)
        else:
            h0 = h0.view(bsz, self.num_blocks, self.block_size)

        if mode == "parallel":
            states = parallel_affine_scan(ops.a, ops.b, h0, kind="block")
        elif mode == "sequential":
            states = sequential_affine_scan(ops.a, ops.b, h0, kind="block")
        else:
            raise ValueError("mode must be 'parallel' or 'sequential'")
        return states.reshape(bsz, injection.shape[1], self.state_dim)

    def stability_bound(self) -> Tensor:
        """Infinity-norm upper bound for each long-term local block."""

        return self.base_blocks().abs().sum(dim=-1).amax(dim=-1)
