from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FlowConfig:
    """Configuration for a scan-friendly FlowCore substrate.

    The current scalable backend uses block-diagonal affine dynamics.  Each block
    is a small local microcircuit; all blocks can be scanned in parallel through
    time because the operator family is closed under composition.
    """

    input_dim: int
    output_dim: int
    state_dim: int = 256
    block_size: int = 8
    num_ports: int = 4
    context_dim: int = 0
    controller_hidden_dim: int = 128
    encoder_hidden_dim: int = 128
    readout_hidden_dim: int = 128
    active_ports: int | None = 2
    stability_scale: float = 0.90
    refinement_steps: int = 1

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.output_dim <= 0 or self.state_dim <= 0:
            raise ValueError("input_dim, output_dim and state_dim must be positive")
        if self.block_size <= 0 or self.state_dim % self.block_size != 0:
            raise ValueError("state_dim must be divisible by block_size")
        if self.num_ports <= 0:
            raise ValueError("num_ports must be positive")
        if self.active_ports is not None:
            if self.active_ports <= 0 or self.active_ports > self.num_ports:
                raise ValueError("active_ports must be in [1, num_ports] or None")
        if not 0.0 < self.stability_scale <= 1.0:
            raise ValueError("stability_scale must be in (0, 1]")
        if self.refinement_steps <= 0:
            raise ValueError("refinement_steps must be positive")

    @property
    def num_blocks(self) -> int:
        return self.state_dim // self.block_size
