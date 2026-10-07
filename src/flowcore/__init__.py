"""FlowCore: parallelizable dynamic substrates built from composable operators."""

from .config import FlowConfig
from .controller import RouteHoldController
from .diagnostics import StateDiagnostics, relative_state_error, state_diagnostics
from .injection import MultiPortInjector, PortSelector
from .memory import PolicyAnchorBuffer
from .model import FlowCoreModel, RefinedFlowCoreModel
from .operators import (
    apply_affine,
    compose_affine,
    make_compiled_scan,
    parallel_affine_scan,
    scan_composition_count,
    sequential_affine_scan,
)
from .substrate import ParallelFlowSubstrate

__all__ = [
    "FlowConfig",
    "FlowCoreModel",
    "RefinedFlowCoreModel",
    "ParallelFlowSubstrate",
    "RouteHoldController",
    "MultiPortInjector",
    "PortSelector",
    "PolicyAnchorBuffer",
    "StateDiagnostics",
    "state_diagnostics",
    "relative_state_error",
    "apply_affine",
    "compose_affine",
    "parallel_affine_scan",
    "make_compiled_scan",
    "scan_composition_count",
    "sequential_affine_scan",
]

__version__ = "0.2.0"
