from __future__ import annotations

import torch
from torch import Tensor, nn


def _topk_simplex(logits: Tensor, k: int | None) -> Tensor:
    """Softmax weights with optional hard top-k support and straight gradients inside support."""

    if k is None or k >= logits.shape[-1]:
        return torch.softmax(logits, dim=-1)
    values, indices = logits.topk(k, dim=-1)
    selected = torch.softmax(values, dim=-1)
    weights = torch.zeros_like(logits)
    return weights.scatter(-1, indices, selected)


class PortSelector(nn.Module):
    """Selects a small number of candidate input ports from encoded content/context."""

    def __init__(self, feature_dim: int, num_ports: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_ports),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class MultiPortInjector(nn.Module):
    """Learned multi-port input map ``sum_k s_k B_k z``.

    Port weights are normalized to sum to one, so increasing the number of ports
    does not trivially increase injection amplitude.  An optional fixed support
    mask can constrain each port to a candidate basin in the global state.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        num_ports: int,
        *,
        active_ports: int | None = None,
        support_mask: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.num_ports = num_ports
        self.active_ports = active_ports
        self.weight = nn.Parameter(torch.empty(num_ports, state_dim, input_dim))
        nn.init.xavier_uniform_(self.weight)

        if support_mask is None:
            support_mask = torch.ones(num_ports, state_dim)
        if support_mask.shape != (num_ports, state_dim):
            raise ValueError("support_mask must have shape [num_ports, state_dim]")
        self.register_buffer("support_mask", support_mask.to(dtype=torch.float32))

    @staticmethod
    def disjoint_support_mask(num_ports: int, state_dim: int) -> Tensor:
        """Create approximately equal, non-overlapping candidate basins."""

        mask = torch.zeros(num_ports, state_dim)
        chunks = torch.tensor_split(torch.arange(state_dim), num_ports)
        for k, idx in enumerate(chunks):
            mask[k, idx] = 1.0
        return mask

    def forward(self, z: Tensor, port_logits: Tensor) -> tuple[Tensor, Tensor]:
        if z.shape[:-1] != port_logits.shape[:-1]:
            raise ValueError("z and port_logits must share leading dimensions")
        if z.shape[-1] != self.input_dim:
            raise ValueError("unexpected input feature dimension")
        if port_logits.shape[-1] != self.num_ports:
            raise ValueError("unexpected port dimension")

        port_weights = _topk_simplex(port_logits, self.active_ports)
        effective_weight = self.weight * self.support_mask[..., None]
        # [..., K, S] = [..., D] x [K, S, D]
        per_port = torch.einsum("...d,ksd->...ks", z, effective_weight)
        injected = torch.einsum("...k,...ks->...s", port_weights, per_port)
        return injected, port_weights
