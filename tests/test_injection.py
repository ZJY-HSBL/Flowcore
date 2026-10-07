import torch

from flowcore.injection import MultiPortInjector


def test_disjoint_support_and_topk_budget():
    torch.manual_seed(4)
    mask = MultiPortInjector.disjoint_support_mask(4, 16)
    injector = MultiPortInjector(6, 16, 4, active_ports=2, support_mask=mask)
    z = torch.randn(2, 7, 6)
    logits = torch.randn(2, 7, 4)
    injected, weights = injector(z, logits)
    assert injected.shape == (2, 7, 16)
    assert weights.shape == (2, 7, 4)
    assert torch.all((weights > 0).sum(dim=-1) == 2)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 7))
