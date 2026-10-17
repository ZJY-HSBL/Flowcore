import torch
import pytest

from flowcore.route_losses import (
    _competing_permutation,
    global_permutation_hinge,
)


def test_global_hinge_uses_best_wrong_full_permutation():
    # Candidate scores: I = 9, transposition (0,1) = 5,
    # all 3-cycles = 0. Therefore competitor swaps 0/1.
    scores = torch.tensor(
        [[[3.0, 1.0, 0.0],
          [1.0, 3.0, 0.0],
          [0.0, 0.0, 3.0]]],
        requires_grad=True,
    )
    gold = torch.tensor([[0, 1, 2]])
    wrong = _competing_permutation(scores, gold)
    torch.testing.assert_close(wrong, torch.tensor([[1, 0, 2]]))
    # gold mean=3, alternate mean=5/3; margin=2 => hinge=2/3
    loss = global_permutation_hinge(scores, gold, margin=2.0)
    torch.testing.assert_close(loss, torch.tensor(2.0/3.0))
    loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_global_hinge_zero_if_full_assignment_gap_sufficient():
    score = torch.tensor(
        [[[8., 0., 0.],
          [0., 8., 0.],
          [0., 0., 8.]]],
        requires_grad=True,
    )
    loss = global_permutation_hinge(score, torch.tensor([[0, 1, 2]]))
    assert loss.item() == 0.0
    loss.backward()
    assert score.grad is not None


def test_global_hinge_matches_bruteforce_for_random_batches():
    import itertools
    torch.manual_seed(143)
    for modules in (3, 4, 5):
        score = torch.randn(2, modules, modules, requires_grad=True)
        gold = torch.stack([torch.randperm(modules) for _ in range(2)])
        candidates = list(itertools.permutations(range(modules)))
        manual = []
        for b in range(2):
            expected = tuple(gold[b].tolist())
            values = [
                sum(score[b, i, perm[i]].item() for i in range(modules))/modules
                for perm in candidates if perm != expected
            ]
            right = score[b, torch.arange(modules), gold[b]].mean()
            manual.append(torch.relu(0.15+right.new_tensor(max(values))-right))
        actual = global_permutation_hinge(score, gold, margin=0.15)
        torch.testing.assert_close(actual, torch.stack(manual).mean())
        grad = torch.autograd.grad(actual, score)[0]
        assert torch.isfinite(grad).all()


def test_global_hinge_validates_permutation_and_size():
    with pytest.raises(ValueError):
        global_permutation_hinge(torch.randn(2, 3, 3), torch.tensor([[0, 1, 1], [0, 2, 1]]))
    with pytest.raises(ValueError):
        global_permutation_hinge(torch.randn(1, 9, 9), torch.arange(9).unsqueeze(0))
    with pytest.raises(ValueError):
        global_permutation_hinge(torch.randn(1, 3, 3), torch.tensor([[0, 1, 2]]), margin=-0.1)
