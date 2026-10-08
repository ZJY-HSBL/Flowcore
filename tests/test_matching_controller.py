import torch

from flowcore.discrete_controller import (
    MatchingPermutationController,
)


def test_matching_controller_shapes_and_valid_hard_assignment():
    torch.manual_seed(70)
    controller = MatchingPermutationController(
        context_dim=5, rank=4
    )
    source = torch.randn(3, 6, 5)
    target = torch.randn(3, 6, 5)
    score = controller.score_matrix(source, target)
    soft = controller.soft_matrix(source, target)
    hard = controller.hard_permutation(source, target)
    assert score.shape == (3, 6, 6)
    assert soft.shape == (3, 6, 6)
    assert hard.shape == (3, 6)
    expected = torch.arange(6)
    for row in hard:
        torch.testing.assert_close(row.sort().values, expected)


def test_matching_controller_identity_projection_can_recover_shuffle():
    torch.manual_seed(71)
    modules = 6
    controller = MatchingPermutationController(
        context_dim=6, rank=6
    )
    with torch.no_grad():
        controller.source_projection.weight.copy_(torch.eye(6))
        controller.target_projection.weight.copy_(torch.eye(6))

    source = torch.eye(6).unsqueeze(0)
    permutation = torch.tensor([[4, 1, 5, 0, 3, 2]])
    target = torch.gather(
        source,
        1,
        permutation.unsqueeze(-1).expand(-1, -1, 6),
    )
    hard = controller.hard_permutation(
        source, target, temperature=0.05
    )
    torch.testing.assert_close(hard, permutation)
