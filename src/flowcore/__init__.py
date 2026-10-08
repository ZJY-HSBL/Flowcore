"""FlowCore: parallelizable dynamic substrates built from composable operators."""

from .circulant import (
    apply_circulant_kronecker_affine,
    compose_circulant_kronecker_affine,
    kernel_to_spectrum,
    parallel_circulant_kronecker_scan,
    sequential_circulant_kronecker_scan,
)
from .circulant_substrate import CirculantKroneckerSubstrate
from .config import FlowConfig
from .controller import RouteHoldController
from .diagnostics import StateDiagnostics, relative_state_error, state_diagnostics
from .discrete_controller import (
    TaskPermutationController,
    geometric_temperature,
    maximum_weight_permutation,
    permutation_matrix_batch,
    sinkhorn_matrix,
)
from .evaluation import (
    TransportBatch,
    TransportMetrics,
    destination_state,
    make_transport_batch,
    make_transport_controls,
    parameter_count,
    transport_loss,
    transport_metrics,
)
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
from .monomial import (
    apply_monomial_affine,
    apply_monomial_linear,
    compose_monomial_affine,
    parallel_monomial_scan,
    sequential_monomial_scan,
)
from .operators import (
    apply_affine,
    compose_affine,
    make_compiled_scan,
    parallel_affine_scan,
    scan_composition_count,
    sequential_affine_scan,
)
from .permutation import (
    PermutationBatch,
    PermutationMetrics,
    arbitrary_permutation_bank,
    block_projection_error,
    circulant_projection_error,
    cyclic_permutation_bank,
    is_cyclic_permutation,
    make_permutation_batch,
    make_permutation_controls,
    permutation_loss,
    permutation_matrix,
    permutation_metrics,
    spectral_projection_error,
)
from .spectral_substrate import SpectralKroneckerSubstrate
from .substrate import ParallelFlowSubstrate

__all__ = [
    "FlowConfig",
    "FlowCoreModel",
    "RefinedFlowCoreModel",
    "ParallelFlowSubstrate",
    "SpectralKroneckerSubstrate",
    "CirculantKroneckerSubstrate",
    "RouteHoldController",
    "TaskPermutationController",
    "MultiPortInjector",
    "PortSelector",
    "PolicyAnchorBuffer",
    "StateDiagnostics",
    "TransportBatch",
    "TransportMetrics",
    "PermutationBatch",
    "PermutationMetrics",
    "state_diagnostics",
    "relative_state_error",
    "make_transport_batch",
    "make_transport_controls",
    "destination_state",
    "transport_loss",
    "transport_metrics",
    "parameter_count",
    "make_permutation_batch",
    "make_permutation_controls",
    "permutation_loss",
    "permutation_metrics",
    "permutation_matrix",
    "cyclic_permutation_bank",
    "arbitrary_permutation_bank",
    "is_cyclic_permutation",
    "block_projection_error",
    "circulant_projection_error",
    "spectral_projection_error",
    "sinkhorn_matrix",
    "maximum_weight_permutation",
    "permutation_matrix_batch",
    "geometric_temperature",
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
    "sequential_spectral_kronecker_scan",
    "kernel_to_spectrum",
    "apply_circulant_kronecker_affine",
    "compose_circulant_kronecker_affine",
    "parallel_circulant_kronecker_scan",
    "sequential_circulant_kronecker_scan",
    "apply_monomial_linear",
    "apply_monomial_affine",
    "compose_monomial_affine",
    "parallel_monomial_scan",
    "sequential_monomial_scan",
]

__version__ = "0.8.0"
