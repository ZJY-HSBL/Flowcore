import torch

from flowcore import (
    CirculantKroneckerSubstrate,
    FlowConfig,
    FlowCoreModel,
    apply_circulant_kronecker_affine,
    kernel_to_spectrum,
    parallel_circulant_kronecker_scan,
    sequential_circulant_kronecker_scan,
)


def _explicit_circular_apply(kernel, state):
    out = torch.zeros_like(state)
    for shift in range(kernel.shape[-1]):
        out = out + kernel[..., shift, None, None] * torch.roll(
            state, shifts=shift, dims=-2
        )
    return out


def test_circulant_apply_matches_explicit_roll():
    torch.manual_seed(30)
    batch, modules, local_dim = 2, 5, 3
    kernel = torch.rand(batch, modules)
    kernel = kernel / kernel.sum(dim=-1, keepdim=True)
    spectrum = kernel_to_spectrum(kernel)
    local = 0.2 * torch.randn(batch, local_dim, local_dim)
    state = torch.randn(batch, modules, local_dim)
    bias = torch.randn(batch, modules, local_dim)

    expected = _explicit_circular_apply(kernel, state)
    expected = torch.matmul(expected, local.transpose(-1, -2)) + bias
    actual = apply_circulant_kronecker_affine(
        spectrum, local, bias, state
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_circulant_parallel_matches_sequential_and_gradients():
    torch.manual_seed(31)
    batch, time, modules, local_dim = 2, 37, 5, 3
    kernel = torch.rand(batch, time, modules)
    kernel = kernel / kernel.sum(dim=-1, keepdim=True)
    base_s = kernel_to_spectrum(kernel)
    eye = torch.eye(local_dim).view(1, 1, local_dim, local_dim)
    base_l = eye + 0.02 * torch.randn(batch, time, local_dim, local_dim)
    base_b = 0.1 * torch.randn(batch, time, modules, local_dim)
    h0 = torch.randn(batch, modules, local_dim)

    seq = sequential_circulant_kronecker_scan(
        base_s, base_l, base_b, h0
    )
    par = parallel_circulant_kronecker_scan(
        base_s, base_l, base_b, h0
    )
    torch.testing.assert_close(seq, par, rtol=8e-5, atol=8e-5)

    kernel1 = kernel[:, :9].clone().requires_grad_(True)
    s1 = kernel_to_spectrum(kernel1)
    l1 = base_l[:, :9].clone().requires_grad_(True)
    b1 = base_b[:, :9].clone().requires_grad_(True)
    loss1 = sequential_circulant_kronecker_scan(
        s1, l1, b1, h0
    ).square().mean()
    grad1 = torch.autograd.grad(loss1, (kernel1, l1, b1))

    kernel2 = kernel[:, :9].clone().requires_grad_(True)
    s2 = kernel_to_spectrum(kernel2)
    l2 = base_l[:, :9].clone().requires_grad_(True)
    b2 = base_b[:, :9].clone().requires_grad_(True)
    loss2 = parallel_circulant_kronecker_scan(
        s2, l2, b2, h0
    ).square().mean()
    grad2 = torch.autograd.grad(loss2, (kernel2, l2, b2))

    torch.testing.assert_close(loss1, loss2, rtol=8e-5, atol=8e-5)
    for left, right in zip(grad1, grad2, strict=True):
        torch.testing.assert_close(left, right, rtol=3e-4, atol=3e-4)


def test_circulant_substrate_has_directional_shift():
    torch.manual_seed(32)
    substrate = CirculantKroneckerSubstrate(
        state_dim=8,
        block_size=2,
        stability_scale=0.95,
    )
    with torch.no_grad():
        substrate.raw_kernel_prior.zero_()

    injection = torch.zeros(1, 3, 8)
    injection[:, 0, :2] = torch.tensor([1.0, -0.5])

    # Offset +1 dominates.  Under the FFT convention used by the implementation,
    # source module 0 moves to physical module 1 on the next transition.
    route = torch.zeros(1, 3, 4)
    route[..., 1] = 1.0
    hold = torch.ones_like(route)

    states = substrate(
        injection, route, hold, mode="parallel"
    ).view(1, 3, 4, 2)
    assert states[:, 1, 1].abs().sum() > states[:, 1, 3].abs().sum()


def test_flow_model_circulant_backend_backward():
    torch.manual_seed(33)
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
        substrate_kind="circulant_kronecker",
    )
    model = FlowCoreModel(cfg)
    x = torch.randn(2, 19, 5)
    context = torch.randn(2, 2)
    par = model(x, context=context, mode="parallel")
    seq = model(x, context=context, mode="sequential")
    torch.testing.assert_close(par.states, seq.states, rtol=1e-4, atol=1e-4)
    loss = par.output.square().mean()
    loss.backward()
    assert model.substrate.raw_local.grad is not None
    assert model.substrate.raw_kernel_prior.grad is not None
