import torch

from flowcore import (
    FlowConfig,
    FlowCoreModel,
    SpectralKroneckerSubstrate,
    apply_spectral_kronecker_affine,
    compose_spectral_kronecker_affine,
    materialize_module_matrix,
    orthogonal_mixing_basis,
    parallel_spectral_kronecker_scan,
    sequential_spectral_kronecker_scan,
)


def test_spectral_apply_matches_materialized_module_matrix():
    torch.manual_seed(20)
    batch, modules, local_dim = 2, 5, 3
    basis = orthogonal_mixing_basis(modules, seed=3)
    gain = torch.rand(batch, modules)
    local = 0.2 * torch.randn(batch, local_dim, local_dim)
    bias = torch.randn(batch, modules, local_dim)
    state = torch.randn(batch, modules, local_dim)

    route_matrix = materialize_module_matrix(gain, basis)
    expected = torch.matmul(route_matrix, state)
    expected = torch.matmul(expected, local.transpose(-1, -2)) + bias
    actual = apply_spectral_kronecker_affine(
        gain, local, bias, state, basis
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_spectral_composition_is_exact():
    torch.manual_seed(21)
    batch, modules, local_dim = 2, 4, 3
    basis = orthogonal_mixing_basis(modules, seed=4)
    x = torch.randn(batch, modules, local_dim)

    g1 = torch.rand(batch, modules)
    l1 = 0.2 * torch.randn(batch, local_dim, local_dim)
    b1 = torch.randn(batch, modules, local_dim)
    g2 = torch.rand(batch, modules)
    l2 = 0.2 * torch.randn(batch, local_dim, local_dim)
    b2 = torch.randn(batch, modules, local_dim)

    direct = apply_spectral_kronecker_affine(
        g2,
        l2,
        b2,
        apply_spectral_kronecker_affine(g1, l1, b1, x, basis),
        basis,
    )
    gc, lc, bc = compose_spectral_kronecker_affine(
        g2, l2, b2, g1, l1, b1, basis
    )
    composed = apply_spectral_kronecker_affine(
        gc, lc, bc, x, basis
    )
    torch.testing.assert_close(direct, composed, rtol=2e-5, atol=2e-5)


def test_spectral_parallel_matches_sequential_and_gradients():
    torch.manual_seed(22)
    batch, time, modules, local_dim = 2, 37, 5, 3
    basis = orthogonal_mixing_basis(modules, seed=5)
    base_g = torch.sigmoid(torch.randn(batch, time, modules))
    eye = torch.eye(local_dim).view(1, 1, local_dim, local_dim)
    base_l = eye + 0.02 * torch.randn(batch, time, local_dim, local_dim)
    base_b = 0.1 * torch.randn(batch, time, modules, local_dim)
    h0 = torch.randn(batch, modules, local_dim)

    seq = sequential_spectral_kronecker_scan(
        base_g, base_l, base_b, h0, basis
    )
    par = parallel_spectral_kronecker_scan(
        base_g, base_l, base_b, h0, basis
    )
    torch.testing.assert_close(seq, par, rtol=5e-5, atol=5e-5)

    g1 = base_g[:, :9].clone().requires_grad_(True)
    l1 = base_l[:, :9].clone().requires_grad_(True)
    b1 = base_b[:, :9].clone().requires_grad_(True)
    loss1 = sequential_spectral_kronecker_scan(
        g1, l1, b1, h0, basis
    ).square().mean()
    grad1 = torch.autograd.grad(loss1, (g1, l1, b1))

    g2 = base_g[:, :9].clone().requires_grad_(True)
    l2 = base_l[:, :9].clone().requires_grad_(True)
    b2 = base_b[:, :9].clone().requires_grad_(True)
    loss2 = parallel_spectral_kronecker_scan(
        g2, l2, b2, h0, basis
    ).square().mean()
    grad2 = torch.autograd.grad(loss2, (g2, l2, b2))

    torch.testing.assert_close(loss1, loss2, rtol=5e-5, atol=5e-5)
    for left, right in zip(grad1, grad2, strict=True):
        torch.testing.assert_close(left, right, rtol=2e-4, atol=2e-4)


def test_spectral_substrate_transfers_signal_between_modules():
    torch.manual_seed(23)
    substrate = SpectralKroneckerSubstrate(
        state_dim=8,
        block_size=2,
        stability_scale=0.95,
        mixing_basis_seed=11,
    )
    batch, time = 1, 4
    injection = torch.zeros(batch, time, 8)
    injection[:, 0, :2] = torch.tensor([1.0, -0.5])
    route = torch.tensor([[[0.10, 0.90, 0.20, 0.80]]]).expand(batch, time, 4)
    hold = torch.ones(batch, time, 4)

    seq = substrate(injection, route, hold, mode="sequential")
    par = substrate(injection, route, hold, mode="parallel")
    torch.testing.assert_close(seq, par, rtol=5e-5, atol=5e-5)

    matrix_state = seq.view(batch, time, 4, 2)
    transferred = matrix_state[:, 1, 1:].abs().sum()
    assert float(transferred) > 1e-5


def test_flow_model_spectral_backend_backward():
    torch.manual_seed(24)
    cfg = FlowConfig(
        input_dim=5,
        output_dim=3,
        state_dim=32,
        block_size=4,
        num_ports=4,
        context_dim=2,
        controller_hidden_dim=24,
        encoder_hidden_dim=16,
        readout_hidden_dim=16,
        active_ports=2,
        substrate_kind="spectral_kronecker",
    )
    model = FlowCoreModel(cfg)
    x = torch.randn(2, 19, 5)
    context = torch.randn(2, 2)
    par = model(x, context=context, mode="parallel")
    seq = model(x, context=context, mode="sequential")
    torch.testing.assert_close(par.states, seq.states, rtol=6e-5, atol=6e-5)
    loss = par.output.square().mean()
    loss.backward()
    assert model.substrate.raw_local.grad is not None
    assert model.substrate.raw_mode_strength.grad is not None
