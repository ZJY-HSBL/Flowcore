"""FlowCore: parallelizable dynamic substrates built from composable operators."""

from .circulant import (\n    apply_circulant_kronecker_affine,\n    compose_circulant_kronecker_affine,\n    kernel_to_spectrum,\n    parallel_circulant_kronecker_scan,\n    sequential_circulant_kronecker_scan,\n)\nfrom .circulant_substrate import CirculantKroneckerSubstrate\nfrom .config import FlowConfig
from .controller import RouteHoldController
from .diagnostics import StateDiagnostics, relative_state_error, state_diagnostics
from .injection import MultiPortInjector, PortSelector
from .kronecker import (
    apply_spectral_kronecker_affine,
    compose_spectral_kronecker_affine,
    materialize_module_matrix,
    orthogonal_mixing_basis,
    parallel_spectral_kronecker_scan,
    sequential_spectral_kronecker_scan,
)
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
from .spectral_substrate import SpectralKroneckerSubstrate
from .substrate import ParallelFlowSubstrate

__all__ = [
    "FlowConfig",
    "FlowCoreModel",
    "RefinedFlowCoreModel",
    "ParallelFlowSubstrate",
    "SpectralKroneckerSubstrate",\n    "CirculantKroneckerSubstrate",
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
    "orthogonal_mixing_basis",
    "materialize_module_matrix",
    "apply_spectral_kronecker_affine",
    "compose_spectral_kronecker_affine",
    "parallel_spectral_kronecker_scan",
    "sequential_spectral_kronecker_scan",\n    "kernel_to_spectrum",\n    "apply_circulant_kronecker_affine",\n    "compose_circulant_kronecker_affine",\n    "parallel_circulant_kronecker_scan",\n    "sequential_circulant_kronecker_scan",
]

__version__ = "0.4.0"
