import torch

from flowcore.discrete_controller import (
    FactorizedTaskPermutationController,
)


def test_factorized_controller_parameter_count_and_shapes():
    torch.manual_seed(60)
    tasks, modules, rank = 4, 6, 2
    controller = FactorizedTaskPermutationController(
        tasks, modules, rank
    )
    expected_params = modules * rank * (tasks + 1)
    actual_params = sum(
        p.numel() for p in controller.parameters()
    )
    assert actual_params == expected_params

    task_id = torch.tensor([0, 2, 3])
    scores = controller.score_matrix(task_id)
    soft = controller.soft_matrix(task_id)
    hard = controller.hard_permutation(task_id)
    assert scores.shape == (3, modules, modules)
    assert soft.shape == (3, modules, modules)
    assert hard.shape == (3, modules)


def test_factorized_hard_rows_are_valid_permutations():
    torch.manual_seed(61)
    controller = FactorizedTaskPermutationController(
        3, 5, 3
    )
    hard = controller.hard_permutation(torch.arange(3))
    expected = torch.arange(5)
    for row in hard:
        torch.testing.assert_close(
            row.sort().values, expected
        )
