import torch

from flowcore.evaluation import (
    destination_state,
    make_transport_batch,
    make_transport_controls,
    transport_loss,
    transport_metrics,
)


def test_transport_batch_injects_only_source_module():
    generator = torch.Generator().manual_seed(1)
    batch = make_transport_batch(
        8, 5, 4, 2, device="cpu", generator=generator
    )
    state = batch.injection.view(8, 5, 4, 2)
    torch.testing.assert_close(state[:, 0, 0], batch.payload)
    assert torch.count_nonzero(state[:, 0, 1:]) == 0
    assert torch.count_nonzero(state[:, 1:]) == 0
    assert torch.all(batch.destination >= 1)


def test_transport_controls_transfer_once_then_hold():
    destination = torch.tensor([1, 3])
    route_transfer = torch.rand(2, 4)
    route, hold = make_transport_controls(
        destination, route_transfer, 7, 4
    )
    torch.testing.assert_close(route[:, 1], route_transfer)
    assert torch.all(hold[:, 0] == 1)
    assert torch.all(hold[:, 1] == 1)
    assert torch.all(hold[:, 2:] == 0)


def test_transport_metrics_reward_exact_delivery():
    batch_size, time, modules, local_dim = 3, 4, 4, 2
    destination = torch.tensor([1, 2, 3])
    payload = torch.randn(batch_size, local_dim)
    states = torch.zeros(batch_size, time, modules * local_dim)
    final = states[:, -1].view(batch_size, modules, local_dim)
    final[torch.arange(batch_size), destination] = payload

    gathered = destination_state(
        states, destination, modules, local_dim
    )
    torch.testing.assert_close(gathered, payload)
    loss = transport_loss(
        states, destination, payload, modules, local_dim
    )
    assert float(loss) == 0.0
    metrics = transport_metrics(
        states, destination, payload, modules, local_dim
    )
    assert metrics.target_mse == 0.0
    assert abs(metrics.delivery_fraction - 1.0) < 1e-6
    assert metrics.leakage_mse == 0.0
