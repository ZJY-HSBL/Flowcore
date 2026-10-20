import torch

from flowcore.graph_routing import (
    GraphConsistencyRouter,
    _swap_candidates,
    graph_assignment_cost,
    normalized_distances,
    refine_geometric_assignment,
)


def test_graph_cost_is_invariant_to_rotations_shifts_scales():
    torch.manual_seed(150)
    latent = torch.randn(4, 6, 5)
    q1, _ = torch.linalg.qr(torch.randn(5, 5))
    q2, _ = torch.linalg.qr(torch.randn(5, 5))
    permutation = torch.stack([torch.randperm(6) for _ in range(4)])
    target = torch.gather(
        latent, 1, permutation.unsqueeze(-1).expand(-1, -1, 5)
    )
    source = latent @ q1 * 1.4 + 2.0
    observed = target @ q2 * 0.8 - 3.0
    cost = graph_assignment_cost(
        normalized_distances(source), normalized_distances(observed),
        permutation,
    )
    assert cost.max().item() < 1e-10


def test_graph_refinement_does_not_increase_cost_and_keeps_bijections():
    torch.manual_seed(151)
    source = normalized_distances(torch.randn(6, 8, 4))
    target = normalized_distances(torch.randn(6, 8, 4))
    perm = torch.stack([torch.randperm(8) for _ in range(6)])
    start_cost = graph_assignment_cost(source, target, perm)
    out, new_cost, confidence = refine_geometric_assignment(
        source, target, perm, reliability_floor=0.15, rounds=5
    )
    # Compare using the SAME reliability-weighted cost function.
    weighted_start = graph_assignment_cost(
        source, target, perm, reliability_floor=0.15
    )
    assert torch.all(new_cost <= weighted_start + 1e-7)
    assert torch.isfinite(confidence).all()
    assert torch.all((confidence >= 0) & (confidence <= 1))
    assert out.shape == (6, 8)
    for row in out:
        torch.testing.assert_close(row.sort().values, torch.arange(8))


def test_swaps_enumerate_all_neighbors():
    perm = torch.tensor([[2, 0, 3, 1]])
    candidates = _swap_candidates(perm)
    assert candidates.shape == (1, 6, 4)
    assert len(set(tuple(row.tolist()) for row in candidates[0])) == 6


def test_graph_router_recovers_perfect_rotation_example():
    torch.manual_seed(152)
    latent = torch.randn(6, 6, 5)
    q1, _ = torch.linalg.qr(torch.randn(5, 5))
    q2, _ = torch.linalg.qr(torch.randn(5, 5))
    permutation = torch.stack([torch.randperm(6) for _ in range(6)])
    target = torch.gather(
        latent, 1, permutation.unsqueeze(-1).expand(-1, -1, 5)
    )
    router = GraphConsistencyRouter(reliability_floor=0.15)
    prediction, confidence = router.route_with_confidence(
        latent @ q1, target @ q2
    )
    torch.testing.assert_close(prediction, permutation)
    assert confidence.min().item() > 0


def test_graph_cost_vectorized_matches_scalar_candidates():
    torch.manual_seed(153)
    source = normalized_distances(torch.randn(3, 5, 6))
    target = normalized_distances(torch.randn(3, 5, 6))
    permutation = torch.stack([torch.randperm(5) for _ in range(3)])
    candidates = _swap_candidates(permutation)
    batch_cost = graph_assignment_cost(
        source, target, candidates, reliability_floor=0.17
    )
    for idx in range(candidates.shape[1]):
        scalar = graph_assignment_cost(
            source, target, candidates[:, idx], reliability_floor=0.17
        )
        torch.testing.assert_close(batch_cost[:, idx], scalar)
