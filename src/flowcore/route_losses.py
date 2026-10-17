"""Losses for learning module assignments from the same payload pairs.

The correspondence targets here are *derived from source and target payloads*,
which are already supplied to the reconstruction objective. They expose
explicit matching supervision that the reconstruction loss only uses
implicitly; this must be disclosed when interpreting comparisons.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


@torch.no_grad()
def infer_payload_correspondence(
    payload: Tensor,
    target_payload: Tensor,
) -> Tensor:
    """Find target-to-source indices by payload distance.

    Input shapes: [batch, modules, payload_dim]. Targets are an exact
    permutation of inputs in the synthetic benchmark. Duplicate payload
    vectors or independently corrupted payloads can make this ambiguous.
    """
    if payload.ndim != 3 or payload.shape != target_payload.shape:
        raise ValueError("payload and target_payload must have the same [B,M,D] shape")
    return torch.cdist(target_payload, payload).argmin(dim=-1)


def bidirectional_hard_negative_margin(
    scores: Tensor,
    correspondence: Tensor,
    *,
    margin: float = 0.1,
) -> Tensor:
    """Target/source symmetric hard-negative hinge on raw assignment scores.

    For every output row i, enforce
        S[i, positive_i] >= max_{j != positive_i} S[i,j] + margin.
    For every source column j, enforce the analogous target-row inequality.

    This controls competition at both ends of the one-to-one assignment.
    The correspondence index is discrete; gradients flow only through scores.
    """
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if scores.ndim != 3 or scores.shape[-1] != scores.shape[-2]:
        raise ValueError("scores must be [batch, modules, modules]")
    batch, modules, _ = scores.shape
    if modules < 2 or correspondence.shape != (batch, modules):
        raise ValueError("correspondence must be [batch, modules] with M>=2")
    if correspondence.dtype != torch.long:
        raise ValueError("correspondence must contain long indices")
    if bool((correspondence < 0).any()) or bool((correspondence >= modules).any()):
        raise ValueError("correspondence indices outside module range")
    if not bool(torch.all(correspondence.sort(-1).values ==
                          torch.arange(modules,device=scores.device))):
        raise ValueError("correspondence must be a bijection")

    positive = scores.gather(dim=2, index=correspondence.unsqueeze(-1)).squeeze(-1)
    is_positive = F.one_hot(correspondence, num_classes=modules).bool()
    row_negative = scores.masked_fill(is_positive, torch.finfo(scores.dtype).min)
    hardest_row = row_negative.amax(dim=-1)
    row_loss = F.relu(margin + hardest_row - positive).mean()

    inverse = correspondence.argsort(dim=-1)
    col_positive = scores.gather(
        dim=1, index=inverse.unsqueeze(1).expand(-1, 1, -1)
    ).squeeze(1)
    col_negative = scores.masked_fill(is_positive, torch.finfo(scores.dtype).min)
    hardest_col = col_negative.amax(dim=1)
    col_loss = F.relu(margin + hardest_col - col_positive).mean()
    return 0.5 * (row_loss + col_loss)
