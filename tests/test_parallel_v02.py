import torch

from flowcore.diagnostics import relative_state_error, state_diagnostics
from flowcore.operators import (
    parallel_affine_scan,
    scan_composition_count,
    sequential_affine_scan,
)


def test_work_efficient_matches_hillis_steele_non_power_of_two():
    torch.manual_seed(11)
    batch, time, blocks, dim = 2, 37, 3, 2
    a = 0.12 * torch.randn(batch, time, blocks, dim, dim)
    b = torch.randn(batch, time, blocks, dim)
    h0 = torch.randn(batch, blocks, dim)
    hs = parallel_affine_scan(a, b, h0, kind="block", algorithm="hillis_steele")
    we = parallel_affine_scan(a, b, h0, kind="block", algorithm="work_efficient")
    torch.testing.assert_close(we, hs, rtol=5e-5, atol=5e-5)


def test_work_efficient_uses_less_algorithmic_work_for_long_power_of_two():
    time = 4096
    work = scan_composition_count(time, "work_efficient")
    hillis = scan_composition_count(time, "hillis_steele")
    assert work < hillis
    assert work == 3 * time - 2


def test_long_horizon_stable_scan_stays_finite_and_close():
    torch.manual_seed(12)
    batch, time, dim = 2, 1025, 16
    a = 0.97 + 0.002 * torch.randn(batch, time, dim)
    b = 0.01 * torch.randn(batch, time, dim)
    h0 = torch.randn(batch, dim)
    seq = sequential_affine_scan(a, b, h0, kind="diagonal")
    par = parallel_affine_scan(a, b, h0, kind="diagonal")
    diag = state_diagnostics(par)
    assert diag.finite_fraction == 1.0
    assert relative_state_error(seq, par) < 2e-5
