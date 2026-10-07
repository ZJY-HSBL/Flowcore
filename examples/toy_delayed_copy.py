"""Minimal delayed-memory demo for FlowCore.

Run:
    python examples/toy_delayed_copy.py
"""

from __future__ import annotations

import torch
from torch import nn

from flowcore import FlowConfig, FlowCoreModel


def make_batch(batch: int, time: int, device: torch.device):
    values = torch.randn(batch, 1, device=device)
    x = torch.zeros(batch, time, 2, device=device)
    x[:, 0, :1] = values
    x[:, 0, 1] = 1.0
    target = values.squeeze(-1)
    return x, target


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = FlowConfig(
        input_dim=2,
        output_dim=1,
        state_dim=64,
        block_size=4,
        num_ports=4,
        context_dim=0,
        active_ports=2,
        controller_hidden_dim=64,
        encoder_hidden_dim=32,
        readout_hidden_dim=32,
    )
    model = FlowCoreModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.MSELoss()

    for step in range(300):
        x, y = make_batch(128, 32, device)
        pred = model(x, mode="parallel", return_sequence=False).output.squeeze(-1)
        loss = loss_fn(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"step={step:03d} loss={loss.item():.6f}")

    x, y = make_batch(1024, 64, device)  # 2x training horizon
    with torch.no_grad():
        pred = model(x, mode="parallel", return_sequence=False).output.squeeze(-1)
        mse = loss_fn(pred, y).item()
    print(f"length-64 MSE: {mse:.6f}")


if __name__ == "__main__":
    main()
