import torch
import pytest

from flowcore.route_losses import (
    bidirectional_hard_negative_margin,
    infer_payload_correspondence,
)


def test_correspondence_is_derived_from_existing_payload_pairs():
    torch.manual_seed(131)
    source = torch.randn(4, 6, 3)
    permutation = torch.stack([torch.randperm(6) for _ in range(4)])
    target = torch.gather(source, 1, permutation.unsqueeze(-1).expand(-1, -1, 3))
    inferred = infer_payload_correspondence(source, target)
    torch.testing.assert_close(inferred, permutation)


def test_margin_is_zero_for_clear_assignment_and_positive_for_violations():
    permutation = torch.tensor([[2, 0, 1]])
    correct = torch.full((1, 3, 3), -1.0)
    correct[0, torch.arange(3), permutation[0]] = 2.0
    assert float(bidirectional_hard_negative_margin(correct, permutation, margin=0.2)) == 0.0
    wrong = -correct
    loss = bidirectional_hard_negative_margin(wrong, permutation, margin=0.2)
    assert loss.item() > 0


def test_bidirectional_margin_gradient_is_finite():
    torch.manual_seed(132)
    scores = torch.randn(2, 6, 6, requires_grad=True)
    assignment = torch.stack([torch.randperm(6) for _ in range(2)])
    loss = bidirectional_hard_negative_margin(scores, assignment)
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_margin_rejects_duplicate_assignment():
    bad = torch.tensor([[0, 0, 2]])
    with pytest.raises(ValueError):
        bidirectional_hard_negative_margin(torch.randn(1, 3, 3), bad)
