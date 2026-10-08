import torch

from flowcore.discrete_controller import (
    TaskPermutationController,
    maximum_weight_permutation,
    permutation_matrix_batch,
    sinkhorn_matrix,
)


def test_sinkhorn_is_nearly_doubly_stochastic():
    torch.manual_seed(50)
    logits = torch.randn(3, 5, 5)
    matrix = sinkhorn_matrix(
        logits, temperature=0.7, iterations=40
    )
    torch.testing.assert_close(
        matrix.sum(dim=-1),
        torch.ones(3, 5),
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        matrix.sum(dim=-2),
        torch.ones(3, 5),
        rtol=1e-4,
        atol=1e-4,
    )


def test_maximum_weight_permutation_finds_known_assignment():
    scores = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 9.0, 0.0],
            [0.0, 8.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 7.0],
        ]
    )
    permutation = maximum_weight_permutation(scores)
    torch.testing.assert_close(
        permutation, torch.tensor([0, 2, 1, 3])
    )


def test_permutation_matrix_batch_matches_gather_convention():
    permutation = torch.tensor(
        [[2, 0, 1], [1, 2, 0]]
    )
    matrix = permutation_matrix_batch(permutation)
    x = torch.tensor(
        [
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]],
        ]
    )
    routed = torch.matmul(matrix, x)
    expected = torch.gather(
        x,
        1,
        permutation.unsqueeze(-1),
    )
    torch.testing.assert_close(routed, expected)


def test_task_controller_hard_output_is_valid_permutation():
    torch.manual_seed(51)
    controller = TaskPermutationController(4, 6)
    task_id = torch.arange(4)
    permutation = controller.hard_permutation(task_id)
    expected = torch.arange(6)
    for row in permutation:
        torch.testing.assert_close(
            row.sort().values, expected
        )
