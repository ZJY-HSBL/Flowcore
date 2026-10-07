import torch

from flowcore.memory import PolicyAnchorBuffer


def test_policy_anchor_buffer_capacity_and_sample():
    torch.manual_seed(7)
    memory = PolicyAnchorBuffer(capacity=8)
    x = torch.randn(4, 3, 5)
    route = torch.randn(4, 3, 6)
    hold = torch.randn(4, 3, 6)
    memory.add(x, route, hold)
    assert len(memory) == 8
    batch = memory.sample(4)
    assert batch.controller_input.shape == (4, 5)
    assert batch.route_target.shape == (4, 6)
    assert batch.hold_target.shape == (4, 6)
