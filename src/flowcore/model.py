from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .circulant_substrate import CirculantKroneckerSubstrate
from .config import FlowConfig
from .controller import ControlSignals, RouteHoldController
from .injection import MultiPortInjector
from .spectral_substrate import SpectralKroneckerSubstrate
from .substrate import ParallelFlowSubstrate


@dataclass(slots=True)
class FlowOutput:
    output: Tensor
    states: Tensor
    controls: ControlSignals
    port_weights: Tensor


class _StateSummarizer(nn.Module):
    def __init__(self, state_dim: int, summary_dim: int) -> None:
        super().__init__()
        self.summary_dim = summary_dim
        if summary_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(state_dim, summary_dim),
                nn.Tanh(),
            )
        else:
            self.proj = None

    def forward(self, states: Tensor) -> Tensor:
        if self.proj is None:
            return states.new_zeros(*states.shape[:-1], 0)
        return self.proj(states)


def _build_substrate(config: FlowConfig) -> nn.Module:
    if config.substrate_kind == "block":
        return ParallelFlowSubstrate(
            state_dim=config.state_dim,
            block_size=config.block_size,
            stability_scale=config.stability_scale,
        )
    if config.substrate_kind == "spectral_kronecker":
        return SpectralKroneckerSubstrate(
            state_dim=config.state_dim,
            block_size=config.block_size,
            stability_scale=config.stability_scale,
            mixing_basis_seed=config.mixing_basis_seed,
        )
    if config.substrate_kind == "circulant_kronecker":
        return CirculantKroneckerSubstrate(
            state_dim=config.state_dim,
            block_size=config.block_size,
            stability_scale=config.stability_scale,
        )
    raise ValueError(f"unsupported substrate_kind: {config.substrate_kind}")


class FlowCoreModel(nn.Module):
    """End-to-end FlowCore prototype with exact scan-friendly dynamics."""

    def __init__(self, config: FlowConfig, *, port_support_mask: Tensor | None = None) -> None:
        super().__init__()
        self.config = config
        latent = config.encoder_hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, latent),
            nn.SiLU(),
            nn.Linear(latent, latent),
        )
        self.controller = RouteHoldController(
            feature_dim=latent,
            context_dim=config.context_dim,
            state_summary_dim=0,
            num_blocks=config.num_blocks,
            num_ports=config.num_ports,
            hidden_dim=config.controller_hidden_dim,
        )
        if port_support_mask is None:
            port_support_mask = MultiPortInjector.disjoint_support_mask(
                config.num_ports, config.state_dim
            )
        self.injector = MultiPortInjector(
            input_dim=latent,
            state_dim=config.state_dim,
            num_ports=config.num_ports,
            active_ports=config.active_ports,
            support_mask=port_support_mask,
        )
        self.substrate = _build_substrate(config)
        self.readout = nn.Sequential(
            nn.Linear(config.state_dim, config.readout_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.readout_hidden_dim, config.output_dim),
        )

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(
        self,
        x: Tensor,
        *,
        context: Tensor | None = None,
        h0: Tensor | None = None,
        mode: str = "parallel",
        return_sequence: bool = True,
    ) -> FlowOutput:
        z = self.encode(x)
        controls = self.controller(z, context=context)
        injection, port_weights = self.injector(z, controls.port_logits)
        states = self.substrate(
            injection,
            controls.route,
            controls.hold,
            h0=h0,
            mode=mode,
        )
        read_states = states if return_sequence else states[:, -1]
        output = self.readout(read_states)
        return FlowOutput(
            output=output,
            states=states,
            controls=controls,
            port_weights=port_weights,
        )


class RefinedFlowCoreModel(nn.Module):
    """Predict-correct FlowCore with a small fixed number of global scan rounds."""

    def __init__(
        self,
        config: FlowConfig,
        *,
        state_summary_dim: int = 32,
        port_support_mask: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.refinement_steps = config.refinement_steps
        latent = config.encoder_hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, latent),
            nn.SiLU(),
            nn.Linear(latent, latent),
        )
        self.summarizer = _StateSummarizer(config.state_dim, state_summary_dim)
        self.controller = RouteHoldController(
            feature_dim=latent,
            context_dim=config.context_dim,
            state_summary_dim=state_summary_dim,
            num_blocks=config.num_blocks,
            num_ports=config.num_ports,
            hidden_dim=config.controller_hidden_dim,
        )
        if port_support_mask is None:
            port_support_mask = MultiPortInjector.disjoint_support_mask(
                config.num_ports, config.state_dim
            )
        self.injector = MultiPortInjector(
            input_dim=latent,
            state_dim=config.state_dim,
            num_ports=config.num_ports,
            active_ports=config.active_ports,
            support_mask=port_support_mask,
        )
        self.substrate = _build_substrate(config)
        self.readout = nn.Sequential(
            nn.Linear(config.state_dim, config.readout_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.readout_hidden_dim, config.output_dim),
        )

    def forward(
        self,
        x: Tensor,
        *,
        context: Tensor | None = None,
        h0: Tensor | None = None,
        mode: str = "parallel",
        return_sequence: bool = True,
    ) -> FlowOutput:
        z = self.encoder(x)
        summary = z.new_zeros(*z.shape[:-1], self.summarizer.summary_dim)
        states = None
        controls = None
        port_weights = None
        for _ in range(self.refinement_steps):
            controls = self.controller(z, context=context, state_summary=summary)
            injection, port_weights = self.injector(z, controls.port_logits)
            states = self.substrate(
                injection,
                controls.route,
                controls.hold,
                h0=h0,
                mode=mode,
            )
            summary = self.summarizer(states)
        assert states is not None and controls is not None and port_weights is not None
        read_states = states if return_sequence else states[:, -1]
        output = self.readout(read_states)
        return FlowOutput(output, states, controls, port_weights)
