"""Dimension-independent geometric fingerprint routing.

A small learned positive diagonal metric operates on quantiles of the sorted
intra-set distance fingerprints.  The feature representation is invariant to
independent rotations/reflections of the source and target observation spaces.
This is an intentionally constrained geometric router, not semantic matching.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .discrete_controller import maximum_weight_permutation, sinkhorn_matrix


def quantile_distance_fingerprints(
    context: Tensor, *, bins: int = 8, normalize_scale: bool = True
) -> Tensor:
    """Return [batch, modules, bins] distance-quantile descriptors.

    Sorted pairwise distances are linearly resampled to a fixed number of bins,
    allowing the same metric parameters to be evaluated at different module
    counts. Orthogonal transforms and translation preserve the descriptors;
    interpolating between different module counts does not guarantee identical
    distributions or permutation recovery.
    """
    if context.ndim != 3:
        raise ValueError("context must be [batch, modules, descriptor_dim]")
    if context.shape[1] < 3:
        raise ValueError("at least three modules required")
    if bins < 2:
        raise ValueError("bins must be at least 2")

    distances = torch.cdist(context, context)
    if normalize_scale:
        denominator = distances.square().mean(
            dim=(1, 2), keepdim=True
        ).sqrt().clamp_min(1e-8)
        distances = distances / denominator
    fingerprints = distances.sort(dim=-1).values
    batch, modules, _ = fingerprints.shape
    return F.interpolate(
        fingerprints.reshape(batch * modules, 1, modules),
        size=bins,
        mode="linear",
        align_corners=True,
    ).reshape(batch, modules, bins)


class QuantileInvariantPermutationController(nn.Module):
    """Metric-learned structural router with hard monomial-compatible output.

    The only learnable parameters are 'bins' non-negative metric weights.
    'learnable=False' creates a matched uniform-weight analytical baseline,
    so the comparison isolates whether task-loss learning improves this
    *same* feature family. Neither version uses task IDs or absolute addresses.
    """

    def __init__(
        self,
        *,
        bins: int = 8,
        learnable: bool = True,
        sinkhorn_iterations: int = 20,
    ) -> None:
        super().__init__()
        if bins < 2:
            raise ValueError("bins must be >= 2")
        if sinkhorn_iterations < 1:
            raise ValueError("sinkhorn_iterations must be positive")
        self.bins = bins
        self.sinkhorn_iterations = sinkhorn_iterations
        # Inverse softplus(1), so initialization is a uniform metric.
        initial = torch.full((bins,), math.log(math.expm1(1.0)))
        if learnable:
            self.raw_weights = nn.Parameter(initial)
        else:
            self.register_buffer("raw_weights", initial)

    def metric_weights(self) -> Tensor:
        weights = F.softplus(self.raw_weights)
        return weights / weights.mean().clamp_min(1e-8)

    def score_matrix(
        self, source_context: Tensor, target_context: Tensor
    ) -> Tensor:
        if source_context.shape != target_context.shape:
            raise ValueError("source and target must match in shape")
        source = quantile_distance_fingerprints(source_context, bins=self.bins)
        target = quantile_distance_fingerprints(target_context, bins=self.bins)
        difference = target[:, :, None, :] - source[:, None, :, :]
        weights = self.metric_weights().view(1, 1, 1, self.bins)
        return -(difference.square() * weights).sum(dim=-1)

    def soft_matrix(
        self, source_context: Tensor, target_context: Tensor, *,
        temperature: float = 0.1
    ) -> Tensor:
        return sinkhorn_matrix(
            self.score_matrix(source_context, target_context),
            temperature=temperature,
            iterations=self.sinkhorn_iterations,
        )

    @torch.no_grad()
    def hard_permutation(
        self, source_context: Tensor, target_context: Tensor
    ) -> Tensor:
        return maximum_weight_permutation(
            self.score_matrix(source_context, target_context)
        )
