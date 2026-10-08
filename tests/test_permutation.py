import torch

from flowcore.permutation import (
    arbitrary_permutation_bank,
    circulant_projection_error,
    cyclic_permutation_bank,
    is_cyclic_permutation,
    make_permutation_batch,
    make_permutation_controls,
    permutation_metrics,
)


def test_cyclic_and_arbitrary_banks_are_separated():
    cyclic = cyclic_permutation_bank(3, 6)
    arbitrary = arbitrary_permutation_bank(8, 6, seed=9)
    assert all(is_cyclic_permutation(row) for row in cyclic)
    assert all(not is_cyclic_permutation(row) for row in arbitrary)


def test_circulant_projection_exact_for_cyclic_not_generic_arbitrary():
    cyclic = cyclic_permutation_bank(1, 6)[0]
    arbitrary = arbitrary_permutation_bank(1, 6, seed=11)[0]
    assert circulant_projection_error(cyclic) < 1e-6
    assert circulant_projection_error(arbitrary) > 0.25


def test_permutation_batch_target_and_slot_metric():
    permutations = torch.tensor(
        [[1, 0, 3, 2], [2, 3, 0, 1]]
    )
    generator = torch.Generator().manual_seed(5)
    batch = make_permutation_batch(
        16,
        5,
        permutations,
        3,
        device="cpu",
        generator=generator,
    )
    states = torch.zeros(16, 5, 12)
    states[:, -1] = batch.target.reshape(16, 12)
    metrics = permutation_metrics(
        states,
        batch.payload,
        batch.target,
        batch.task_id,
        permutations,
    )
    assert metrics.mse == 0.0
    assert metrics.slot_accuracy == 1.0


def test_permutation_controls_apply_once_then_persist():
    route_at_transfer = torch.rand(3, 5)
    route, hold = make_permutation_controls(
        route_at_transfer, 9
    )
    torch.testing.assert_close(route[:, 1], route_at_transfer)
    assert torch.all(hold[:, :2] == 1)
    assert torch.all(hold[:, 2:] == 0)
