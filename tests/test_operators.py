import torch

from flowcore.operators import parallel_affine_scan, sequential_affine_scan


def test_dense_parallel_matches_sequential():
    torch.manual_seed(0)
    b, t, d = 3, 17, 5
    a = 0.15 * torch.randn(b, t, d, d)
    bias = torch.randn(b, t, d)
    h0 = torch.randn(b, d)
    seq = sequential_affine_scan(a, bias, h0, kind="dense")
    par = parallel_affine_scan(a, bias, h0, kind="dense")
    torch.testing.assert_close(seq, par, rtol=2e-5, atol=2e-5)


def test_block_parallel_matches_sequential():
    torch.manual_seed(1)
    b, t, k, d = 2, 19, 4, 3
    a = 0.2 * torch.randn(b, t, k, d, d)
    bias = torch.randn(b, t, k, d)
    h0 = torch.randn(b, k, d)
    seq = sequential_affine_scan(a, bias, h0, kind="block")
    par = parallel_affine_scan(a, bias, h0, kind="block")
    torch.testing.assert_close(seq, par, rtol=3e-5, atol=3e-5)


def test_diagonal_parallel_matches_sequential():
    torch.manual_seed(2)
    b, t, d = 4, 33, 11
    a = torch.sigmoid(torch.randn(b, t, d))
    bias = torch.randn(b, t, d)
    h0 = torch.randn(b, d)
    seq = sequential_affine_scan(a, bias, h0, kind="diagonal")
    par = parallel_affine_scan(a, bias, h0, kind="diagonal")
    torch.testing.assert_close(seq, par, rtol=2e-5, atol=2e-5)
