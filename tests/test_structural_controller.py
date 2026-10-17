import torch

from flowcore.structural_controller import (
    QuantileInvariantPermutationController,
    quantile_distance_fingerprints,
)


def test_fingerprints_invariant_to_independent_rotations_and_translations():
    torch.manual_seed(121)
    latent = torch.randn(3, 7, 6)
    rotation, _ = torch.linalg.qr(torch.randn(3, 6, 6))
    shifted = torch.bmm(latent, rotation) + torch.randn(3, 1, 6)
    lhs = quantile_distance_fingerprints(latent)
    rhs = quantile_distance_fingerprints(shifted)
    torch.testing.assert_close(lhs, rhs, rtol=1e-4, atol=1e-4)


def test_structural_controller_recovers_fresh_permutations_two_module_counts():
    torch.manual_seed(122)
    model = QuantileInvariantPermutationController(bins=8, learnable=False)
    for modules in (6, 8):
        source = torch.randn(3, modules, 6)
        perm = torch.stack([torch.randperm(modules) for _ in range(3)])
        q1, _ = torch.linalg.qr(torch.randn(3, 6, 6))
        q2, _ = torch.linalg.qr(torch.randn(3, 6, 6))
        target_latent = torch.gather(
            source, 1, perm.unsqueeze(-1).expand(-1, -1, 6)
        )
        src = torch.bmm(source, q1)
        tgt = torch.bmm(target_latent, q2)
        actual = model.hard_permutation(src, tgt)
        torch.testing.assert_close(actual, perm)


def test_controller_has_eight_metric_parameters_with_trainable_gradients():
    torch.manual_seed(123)
    model = QuantileInvariantPermutationController(bins=8, learnable=True)
    baseline = QuantileInvariantPermutationController(bins=8, learnable=False)
    assert sum(p.numel() for p in model.parameters()) == 8
    assert sum(p.numel() for p in baseline.parameters()) == 0
    source = torch.randn(3, 6, 6)
    target = torch.randn(3, 6, 6)
    scores = model.score_matrix(source, target)
    soft = model.soft_matrix(source, target, temperature=0.2)
    assert scores.shape == (3, 6, 6)
    assert soft.shape == (3, 6, 6)
    ((soft - torch.eye(6)).square().mean()).backward()
    assert model.raw_weights.grad is not None
    assert torch.isfinite(model.raw_weights.grad).all()


def test_structural_controller_rejects_invalid_inputs():
    model = QuantileInvariantPermutationController()
    with torch.no_grad():
        try:
            model.score_matrix(torch.randn(2, 6, 5), torch.randn(2, 7, 5))
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched shapes should be rejected")
