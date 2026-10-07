from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class ControlSignals:
    route: Tensor
    hold: Tensor
    port_logits: Tensor


class RouteHoldController(nn.Module):
    """Small nonlinear controller that emits scan-friendly operator controls.

    The exact parallel path intentionally does not require the current recurrent
    state.  State-conditioned control is supported through an optional summary
    produced by a previous refinement pass.
    """

    def __init__(
        self,
        feature_dim: int,
        context_dim: int,
        state_summary_dim: int,
        num_blocks: int,
        num_ports: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.state_summary_dim = state_summary_dim
        self.num_blocks = num_blocks
        self.num_ports = num_ports
        total_in = feature_dim + context_dim + state_summary_dim
        self.backbone = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.route_head = nn.Linear(hidden_dim, num_blocks)
        self.hold_head = nn.Linear(hidden_dim, num_blocks)
        self.port_head = nn.Linear(hidden_dim, num_ports)

        # Neutral-ish initialization: routes partially open, moderate update rate.
        nn.init.zeros_(self.route_head.weight)
        nn.init.zeros_(self.hold_head.weight)
        nn.init.zeros_(self.port_head.weight)
        nn.init.constant_(self.route_head.bias, 1.0)
        nn.init.constant_(self.hold_head.bias, -2.0)
        nn.init.zeros_(self.port_head.bias)

    def _broadcast_context(self, features: Tensor, context: Tensor | None) -> Tensor:
        leading = features.shape[:-1]
        if self.context_dim == 0:
            return features.new_zeros(*leading, 0)
        if context is None:
            return features.new_zeros(*leading, self.context_dim)
        if context.shape[-1] != self.context_dim:
            raise ValueError("unexpected context_dim")
        if context.ndim == features.ndim - 1:
            context = context.unsqueeze(-2).expand(*leading, self.context_dim)
        if context.shape[:-1] != leading:
            raise ValueError("context must be [B,C] or [B,T,C] for sequence features")
        return context

    def forward(
        self,
        features: Tensor,
        *,
        context: Tensor | None = None,
        state_summary: Tensor | None = None,
    ) -> ControlSignals:
        ctx = self._broadcast_context(features, context)
        if self.state_summary_dim == 0:
            summary = features.new_zeros(*features.shape[:-1], 0)
        elif state_summary is None:
            summary = features.new_zeros(*features.shape[:-1], self.state_summary_dim)
        else:
            if state_summary.shape[:-1] != features.shape[:-1]:
                raise ValueError("state_summary leading dimensions must match features")
            if state_summary.shape[-1] != self.state_summary_dim:
                raise ValueError("unexpected state_summary dimension")
            summary = state_summary

        x = torch.cat((features, ctx, summary), dim=-1)
        h = self.backbone(x)
        route = torch.sigmoid(self.route_head(h))
        # hold is interpreted as update-rate alpha: small => persistent state.
        hold = torch.sigmoid(self.hold_head(h))
        port_logits = self.port_head(h)
        return ControlSignals(route=route, hold=hold, port_logits=port_logits)
