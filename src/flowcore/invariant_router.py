"""Geometry-invariant permutation matching across unknown orthogonal views.

This is a deterministic analytical baseline, not a learned semantic controller.
It uses each node's sorted distances to other nodes as an orthogonal-invariant
fingerprint, then solves a one-to-one assignment.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .discrete_controller import maximum_weight_permutation, sinkhorn_matrix


def distance_fingerprints(context: Tensor, *, normalize_scale: bool = True) -> Tensor:
    """Sorted within-set Euclidean distances for every module.

    A set of points keeps these fingerprints under arbitrary shared orthogonal
    transformations, translations, and (if normalized) positive global scale.
    Fingerprints may be ambiguous in symmetric/degenerate configurations.
    """

    if context.ndim < 3:
        raise ValueError("context must have [..., modules, descriptor_dim]")
    if context.shape[-2] < 3:
        raise ValueError("need at least three modules for robust distance fingerprints")
    distances = torch.cdist(context, context)
    if normalize_scale:
        scale = distances.square().mean(
            dim=(-2, -1), keepdim=True
        ).sqrt().clamp_min(1e-8)
        distances = distances / scale
    return distances.sort(dim=-1).values


def invariant_matching_scores(
    source_context: Tensor,
    target_context: Tensor,
    *,
    normalize_scale: bool = True,
) -> Tensor:
    """Pairwise negative squared distance between invariant fingerprints.

    Output convention is scores[..., target_slot, source_slot].
    """

    if source_context.shape != target_context.shape:
        raise ValueError("source and target contexts must have identical shapes")
    source = distance_fingerprints(source_context, normalize_scale=normalize_scale)
    target = distance_fingerprints(target_context, normalize_scale=normalize_scale)
    return -torch.cdist(target, source).square()


class InvariantPermutationController(nn.Module):
    """Exact hard assignment using orthogonal-invariant structural features.

    Zero trainable parameters.  Complexity remains quadratic in module count
    for the score matrix, followed by the small-M reference assignment solver.
    """

    def __init__(self, *, normalize_scale: bool = True) -> None:
        super().__init__()
        self.normalize_scale = normalize_scale

    def score_matrix(self, source_context: Tensor, target_context: Tensor) -> Tensor:
        return invariant_matching_scores(
            source_context, target_context, normalize_scale=self.normalize_scale
        )

    def soft_matrix(
        self, source_context: Tensor, target_context: Tensor, *, temperature: float = 0.1
    ) -> Tensor:
        return sinkhorn_matrix(
            self.score_matrix(source_context, target_context), temperature=temperature
        )

    @torch.no_grad()
    def hard_permutation(self, source_context: Tensor, target_context: Tensor) -> Tensor:
        return maximum_weight_permutation(
            self.score_matrix(source_context, target_context)
        )
