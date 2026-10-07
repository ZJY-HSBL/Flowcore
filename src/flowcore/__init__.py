"""FlowCore: parallelizable dynamic substrates built from composable operators."""

from .config import FlowConfig
from .controller import RouteHoldController
from .injection import MultiPortInjector, PortSelector
from .memory import PolicyAnchorBuffer
from .model import FlowCoreModel, RefinedFlowCoreModel
from .operators import (
    apply_affine,
    compose_affine,
    parallel_affine_scan,
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
    "apply_affine",
    "compose_affine",
    "parallel_affine_scan",
    "sequential_affine_scan",
]

__version__ = "0.1.0"
