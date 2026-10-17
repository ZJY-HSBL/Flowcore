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



@torch.no_grad()
def _competing_permutation(
    scores: Tensor,
    correspondence: Tensor,
    *,
    max_modules: int = 8,
) -> Tensor:
    """Find the highest-scoring *wrong* whole permutation.

    Exhaustive small-M oracle: O(B * M! * M) matching operations. No gradients
    through the discrete argmax. This is a research reference, NOT an efficient
    large-M assignment solver.
    """
    from itertools import permutations

    if scores.ndim != 3 or scores.shape[-1] != scores.shape[-2]:
        raise ValueError("scores must be [batch, modules, modules]")
    batch, modules, _ = scores.shape
    if modules < 2 or modules > max_modules:
        raise ValueError(f"requires 2 <= modules <= {max_modules}")
    if correspondence.shape != (batch, modules) or correspondence.dtype != torch.long:
        raise ValueError("correspondence must be [batch, modules] long")
    expected = torch.arange(modules, device=scores.device)
    if not bool(torch.all(correspondence.sort(dim=-1).values == expected)):
        raise ValueError("correspondence must be a permutation")

    candidates = torch.tensor(
        list(permutations(range(modules))), dtype=torch.long, device=scores.device
    )
    row = expected.unsqueeze(0)
    # [B, M!, M]; exclude the gold permutation to force an actual competitor.
    selected = scores.detach()[:, row, candidates]
    total = selected.sum(dim=-1)
    is_gold = (candidates.unsqueeze(0) == correspondence.unsqueeze(1)).all(-1)
    total = total.masked_fill(is_gold, float("-inf"))
    index = total.argmax(-1)
    return candidates[index]


def global_permutation_hinge(
    scores: Tensor,
    correspondence: Tensor,
    *,
    margin: float = 0.1,
    max_modules: int = 8,
) -> Tensor:
    """Structured max-margin loss over entire one-to-one assignments.

    L = mean_b relu(margin + score(best_wrong_perm) - score(gold_perm))
    where each permutation score is the *average* of its M selected row
    scores. Discrete competitor selection is detached, gradients flow through
    the selected candidate and gold scores. The fixed hinge margin uses the
    same numeric value as the local objective, but has different semantics.

    This tests a global assignment margin on SMALL module counts; it does not
    accelerate matching and requires correspondences that the experiment
    derives from already-observed paired payloads.
    """
    if margin < 0:
        raise ValueError("margin must be non-negative")
    wrong = _competing_permutation(
        scores, correspondence, max_modules=max_modules
    )
    correct_score = scores.gather(
        2, correspondence.unsqueeze(-1)
    ).squeeze(-1).mean(dim=-1)
    wrong_score = scores.gather(
        2, wrong.unsqueeze(-1)
    ).squeeze(-1).mean(dim=-1)
    return F.relu(margin + wrong_score - correct_score).mean()
