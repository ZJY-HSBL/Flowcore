"""Noise-robust geometric graph matching and selective routing diagnostics.

Start from the zero-parameter sorted-distance fingerprint assignment. Refine
the *whole* permutation using local 2-opt swaps against full pairwise geometry.

All operations are invariant to independent rigid observation transforms and
positive global scale. Optional distance-reliability weights downweight short
edges, where relative geometric perturbation can be large. The reliability
floor is fixed across all test conditions; no oracle noise sigma is supplied.

The returned gap confidence is a *ranking heuristic*, not a calibrated
probability of route correctness. Assignment remains small-M and discrete.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .discrete_controller import maximum_weight_permutation
from .invariant_router import invariant_matching_scores


def normalized_distances(context: Tensor) -> Tensor:
    if context.ndim != 3 or context.shape[1] < 3:
        raise ValueError("context must be [batch, M>=3, descriptor_dim]")
    distances = torch.cdist(context, context)
    return distances / distances.square().mean(
        dim=(-2, -1), keepdim=True
    ).sqrt().clamp_min(1e-8)


def graph_assignment_cost(
    source_distances: Tensor,
    target_distances: Tensor,
    permutations: Tensor,
    *,
    reliability_floor: float = 0.0,
) -> Tensor:
    """Graph distortion for one or many target->source permutations.

    Inputs [B,M,M], [B,M,M], and [B,M] or [B,K,M].
    Returns [B] or [B,K]. Off-diagonal pairs only. The score penalizes
    pairwise distance disagreement, not individual fingerprint differences.
    """
    if reliability_floor < 0:
        raise ValueError("reliability_floor must be nonnegative")
    if (
        source_distances.ndim != 3
        or target_distances.shape != source_distances.shape
        or source_distances.shape[-1] != source_distances.shape[-2]
    ):
        raise ValueError("distance matrices must be matching [B,M,M]")
    original_ndim = permutations.ndim
    if original_ndim not in (2, 3):
        raise ValueError("permutations must be [B,M] or [B,K,M]")
    if original_ndim == 2:
        permutations = permutations.unsqueeze(1)

    batch, modules, _ = source_distances.shape
    if (
        permutations.shape[0] != batch
        or permutations.shape[-1] != modules
        or permutations.dtype != torch.long
    ):
        raise ValueError("permutation shape or dtype mismatch")
    choices = permutations.shape[1]
    indexed_source = source_distances.unsqueeze(1).expand(
        -1, choices, -1, -1
    )
    indexed_source = indexed_source.gather(
        2, permutations.unsqueeze(-1).expand(-1, -1, -1, modules)
    )
    indexed_source = indexed_source.gather(
        3, permutations.unsqueeze(-2).expand(-1, -1, modules, -1)
    )
    target = target_distances.unsqueeze(1)
    error = (indexed_source - target).square()

    if reliability_floor:
        # Smooth deterministic attenuation for short, noise-sensitive edges.
        # This is a fixed heuristic, not a noise variance estimated from labels.
        floor2 = reliability_floor ** 2
        reliability_s = indexed_source.square() / (
            indexed_source.square() + floor2
        )
        reliability_t = target.square() / (target.square() + floor2)
        error = error * (0.5 * (reliability_s + reliability_t))

    diagonal = torch.eye(
        modules, device=source_distances.device, dtype=torch.bool
    ).view(1, 1, modules, modules)
    cost = error.masked_fill(diagonal, 0).sum(dim=(-2, -1)) / (
        modules * (modules - 1)
    )
    return cost.squeeze(1) if original_ndim == 2 else cost


def _swap_candidates(permutation: Tensor) -> Tensor:
    """All one-transposition neighbors, [B, M*(M-1)/2, M]."""
    batch, modules = permutation.shape
    pairs = torch.combinations(
        torch.arange(modules, device=permutation.device), r=2
    )
    variants = permutation.unsqueeze(1).expand(
        batch, pairs.shape[0], modules
    ).clone()
    ids = torch.arange(pairs.shape[0], device=permutation.device)
    a, b = pairs[:, 0], pairs[:, 1]
    variants[:, ids, a] = permutation[:, b]
    variants[:, ids, b] = permutation[:, a]
    return variants


def refine_geometric_assignment(
    source_distances: Tensor,
    target_distances: Tensor,
    initial_permutation: Tensor,
    *,
    reliability_floor: float = 0.0,
    rounds: int = 5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Monotone local swap optimization; return route, cost, confidence.

    At each round the best improving two-node swap is applied. Any positive
    confidence is a normalized gap to the closest one-swap alternative.
    Local optimality does not imply globally correct matching.
    """
    if rounds < 0:
        raise ValueError("rounds must be nonnegative")
    if initial_permutation.ndim != 2:
        raise ValueError("initial_permutation must be [B,M]")
    current = initial_permutation.clone()
    current_cost = graph_assignment_cost(
        source_distances, target_distances, current,
        reliability_floor=reliability_floor,
    )
    for _ in range(rounds):
        candidates = _swap_candidates(current)
        costs = graph_assignment_cost(
            source_distances, target_distances, candidates,
            reliability_floor=reliability_floor,
        )
        best_cost, index = costs.min(dim=1)
        replacement = candidates.gather(
            1, index[:, None, None].expand(-1, 1, current.shape[-1])
        ).squeeze(1)
        improves = best_cost < current_cost - 1e-9
        if not bool(improves.any()):
            break
        current = torch.where(improves[:, None], replacement, current)
        current_cost = torch.where(improves, best_cost, current_cost)

    neighbor_costs = graph_assignment_cost(
        source_distances, target_distances, _swap_candidates(current),
        reliability_floor=reliability_floor,
    )
    gap = (neighbor_costs.min(dim=1).values - current_cost).clamp_min(0)
    confidence = gap / (gap + current_cost + 1e-8)
    return current, current_cost, confidence


class GraphConsistencyRouter(nn.Module):
    """A zero-learned-parameter, geometry-consistent 2-opt controller."""

    def __init__(
        self, *, reliability_floor: float = 0.0, rounds: int = 5
    ) -> None:
        super().__init__()
        if reliability_floor < 0 or rounds < 0:
            raise ValueError("reliability_floor and rounds must be nonnegative")
        self.reliability_floor = reliability_floor
        self.rounds = rounds

    @torch.no_grad()
    def route_with_confidence(
        self, source_context: Tensor, target_context: Tensor
    ) -> tuple[Tensor, Tensor]:
        if source_context.shape != target_context.shape:
            raise ValueError("source and target contexts must match")
        start = maximum_weight_permutation(
            invariant_matching_scores(source_context, target_context)
        )
        source = normalized_distances(source_context)
        target = normalized_distances(target_context)
        result, _, confidence = refine_geometric_assignment(
            source, target, start,
            reliability_floor=self.reliability_floor,
            rounds=self.rounds,
        )
        return result, confidence

    @torch.no_grad()
    def hard_permutation(
        self, source_context: Tensor, target_context: Tensor
    ) -> Tensor:
        return self.route_with_confidence(
            source_context, target_context
        )[0]
