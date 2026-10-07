import torch

from flowcore.operators import parallel_affine_scan, sequential_affine_scan


def test_parallel_and_sequential_gradients_match():
    torch.manual_seed(3)
    b, t, k, d = 2, 9, 3, 2
    base_a = 0.1 * torch.randn(b, t, k, d, d)
    base_bias = torch.randn(b, t, k, d)
    h0 = torch.randn(b, k, d)

    a1 = base_a.clone().requires_grad_(True)
    b1 = base_bias.clone().requires_grad_(True)
    loss1 = sequential_affine_scan(a1, b1, h0, kind="block").square().mean()
    g1 = torch.autograd.grad(loss1, (a1, b1))

    a2 = base_a.clone().requires_grad_(True)
    b2 = base_bias.clone().requires_grad_(True)
    loss2 = parallel_affine_scan(a2, b2, h0, kind="block").square().mean()
    g2 = torch.autograd.grad(loss2, (a2, b2))

    torch.testing.assert_close(loss1, loss2, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(g1[0], g2[0], rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(g1[1], g2[1], rtol=2e-4, atol=2e-4)
