from __future__ import annotations

from dataclasses import dataclass
import random

import torch
from torch import Tensor


@dataclass(slots=True)
class PolicyAnchorBatch:
    controller_input: Tensor
    route_target: Tensor
    hold_target: Tensor


class PolicyAnchorBuffer:
    """Small reservoir-sampled memory for Route/Hold policy anchors.

    This deliberately stores controller-level behavior rather than raw episodes.
    It is a primitive building block for the policy-replay experiments motivated
    by the BetterNN/Flow continual-learning results.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: list[tuple[Tensor, Tensor, Tensor]] = []
        self._seen = 0

    def __len__(self) -> int:
        return len(self._items)

    @torch.no_grad()
    def add(self, controller_input: Tensor, route: Tensor, hold: Tensor) -> None:
        flat_in = controller_input.detach().cpu().reshape(-1, controller_input.shape[-1])
        flat_r = route.detach().cpu().reshape(-1, route.shape[-1])
        flat_h = hold.detach().cpu().reshape(-1, hold.shape[-1])
        for x, r, h in zip(flat_in, flat_r, flat_h, strict=True):
            self._seen += 1
            item = (x.clone(), r.clone(), h.clone())
            if len(self._items) < self.capacity:
                self._items.append(item)
            else:
                j = random.randrange(self._seen)
                if j < self.capacity:
                    self._items[j] = item

    def sample(self, batch_size: int, *, device: torch.device | str | None = None) -> PolicyAnchorBatch:
        if not self._items:
            raise RuntimeError("cannot sample an empty PolicyAnchorBuffer")
        k = min(batch_size, len(self._items))
        selected = random.sample(self._items, k)
        x = torch.stack([item[0] for item in selected])
        r = torch.stack([item[1] for item in selected])
        h = torch.stack([item[2] for item in selected])
        if device is not None:
            x, r, h = x.to(device), r.to(device), h.to(device)
        return PolicyAnchorBatch(x, r, h)
