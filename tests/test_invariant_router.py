import torch

from flowcore.invariant_router import (
    InvariantPermutationController,
    distance_fingerprints,
    invariant_matching_scores,
)


def test_distance_fingerprints_invariant_under_orthogonal_views():
    torch.manual_seed(101)
    points = torch.randn(3, 7, 6)
    q1, _ = torch.linalg.qr(torch.randn(6, 6))
    q2, _ = torch.linalg.qr(torch.randn(6, 6))
    source = points @ q1
    target = points @ q2
    a = distance_fingerprints(source)
    b = distance_fingerprints(target)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_invariant_controller_recovers_unseen_shuffled_views():
    torch.manual_seed(102)
    modules, dim = 6, 6
    latent = torch.randn(4, modules, dim)
    q1, _ = torch.linalg.qr(torch.randn(dim, dim))
    q2, _ = torch.linalg.qr(torch.randn(dim, dim))
    permutation = torch.stack([torch.randperm(modules) for _ in range(4)])
    source = latent @ q1
    target = torch.gather(
        latent,
        1,
        permutation.unsqueeze(-1).expand(-1, -1, dim),
    ) @ q2
    controller = InvariantPermutationController()
    predictions = controller.hard_permutation(source, target)
    torch.testing.assert_close(predictions, permutation)
    scores = invariant_matching_scores(source, target)
    assert scores.shape == (4, modules, modules)


def test_invariant_controller_rejects_shape_mismatch():
    controller = InvariantPermutationController()
    source = torch.randn(2, 6, 5)
    target = torch.randn(2, 7, 5)
    try:
        controller.score_matrix(source, target)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
