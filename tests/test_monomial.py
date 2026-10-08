import torch

from flowcore import (
    apply_monomial_affine,
    arbitrary_permutation_bank,
    compose_monomial_affine,
    make_permutation_batch,
    parallel_monomial_scan,
    permutation_metrics,
    sequential_monomial_scan,
)


def test_monomial_apply_matches_explicit_gather():
    torch.manual_seed(40)
    batch, modules, local_dim = 2, 5, 3
    permutation = torch.stack(
        [torch.randperm(modules), torch.randperm(modules)]
    )
    gain = torch.rand(batch, modules)
    local = torch.randn(batch, local_dim, local_dim) * 0.2
    bias = torch.randn(batch, modules, local_dim)
    state = torch.randn(batch, modules, local_dim)

    index = permutation.unsqueeze(-1).expand(-1, -1, local_dim)
    expected = torch.gather(state, 1, index)
    expected = expected * gain.unsqueeze(-1)
    expected = torch.matmul(
        expected, local.transpose(-1, -2)
    ) + bias

    actual = apply_monomial_affine(
        permutation, gain, local, bias, state
    )
    torch.testing.assert_close(actual, expected)


def test_monomial_composition_is_exact():
    torch.manual_seed(41)
    batch, modules, local_dim = 2, 5, 2
    p1 = torch.stack(
        [torch.randperm(modules), torch.randperm(modules)]
    )
    p2 = torch.stack(
        [torch.randperm(modules), torch.randperm(modules)]
    )
    g1 = torch.rand(batch, modules)
    g2 = torch.rand(batch, modules)
    l1 = torch.randn(batch, local_dim, local_dim) * 0.2
    l2 = torch.randn(batch, local_dim, local_dim) * 0.2
    b1 = torch.randn(batch, modules, local_dim)
    b2 = torch.randn(batch, modules, local_dim)
    x = torch.randn(batch, modules, local_dim)

    direct = apply_monomial_affine(
        p2,
        g2,
        l2,
        b2,
        apply_monomial_affine(p1, g1, l1, b1, x),
    )
    pc, gc, lc, bc = compose_monomial_affine(
        p2, g2, l2, b2, p1, g1, l1, b1
    )
    composed = apply_monomial_affine(
        pc, gc, lc, bc, x
    )
    torch.testing.assert_close(
        direct, composed, rtol=2e-5, atol=2e-5
    )


def test_monomial_parallel_matches_sequential_and_gradients():
    torch.manual_seed(42)
    batch, time, modules, local_dim = 2, 19, 5, 3
    permutation = torch.stack(
        [
            torch.stack([torch.randperm(modules) for _ in range(time)])
            for _ in range(batch)
        ]
    )
    base_gain = 0.8 + 0.1 * torch.rand(batch, time, modules)
    eye = torch.eye(local_dim).view(1, 1, local_dim, local_dim)
    base_local = eye + 0.02 * torch.randn(
        batch, time, local_dim, local_dim
    )
    base_bias = 0.1 * torch.randn(
        batch, time, modules, local_dim
    )
    h0 = torch.randn(batch, modules, local_dim)

    seq = sequential_monomial_scan(
        permutation, base_gain, base_local, base_bias, h0
    )
    par = parallel_monomial_scan(
        permutation, base_gain, base_local, base_bias, h0
    )
    torch.testing.assert_close(
        seq, par, rtol=8e-5, atol=8e-5
    )

    g1 = base_gain[:, :9].clone().requires_grad_(True)
    l1 = base_local[:, :9].clone().requires_grad_(True)
    b1 = base_bias[:, :9].clone().requires_grad_(True)
    loss1 = sequential_monomial_scan(
        permutation[:, :9], g1, l1, b1, h0
    ).square().mean()
    grad1 = torch.autograd.grad(loss1, (g1, l1, b1))

    g2 = base_gain[:, :9].clone().requires_grad_(True)
    l2 = base_local[:, :9].clone().requires_grad_(True)
    b2 = base_bias[:, :9].clone().requires_grad_(True)
    loss2 = parallel_monomial_scan(
        permutation[:, :9], g2, l2, b2, h0
    ).square().mean()
    grad2 = torch.autograd.grad(loss2, (g2, l2, b2))

    torch.testing.assert_close(
        loss1, loss2, rtol=8e-5, atol=8e-5
    )
    for left, right in zip(grad1, grad2, strict=True):
        torch.testing.assert_close(
            left, right, rtol=4e-4, atol=4e-4
        )


def test_monomial_oracle_routes_arbitrary_permutation_exactly():
    torch.manual_seed(43)
    modules, local_dim, time, batch_size = 6, 3, 8, 32
    permutations = arbitrary_permutation_bank(
        4, modules, seed=123
    )
    generator = torch.Generator().manual_seed(7)
    batch = make_permutation_batch(
        batch_size,
        time,
        permutations,
        local_dim,
        device="cpu",
        generator=generator,
    )

    identity = torch.arange(modules).view(1, 1, modules)
    operator_permutation = identity.expand(
        batch_size, time, modules
    ).clone()
    operator_permutation[:, 1] = permutations[batch.task_id]
    gain = torch.ones(batch_size, time, modules)
    local = torch.eye(local_dim).view(
        1, 1, local_dim, local_dim
    ).expand(batch_size, time, local_dim, local_dim)
    bias = batch.injection.view(
        batch_size, time, modules, local_dim
    )
    h0 = torch.zeros(batch_size, modules, local_dim)

    states = parallel_monomial_scan(
        operator_permutation, gain, local, bias, h0
    )
    flat = states.reshape(
        batch_size, time, modules * local_dim
    )
    metrics = permutation_metrics(
        flat,
        batch.payload,
        batch.target,
        batch.task_id,
        permutations,
    )
    assert metrics.mse < 1e-12
    assert metrics.slot_accuracy == 1.0
