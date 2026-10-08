from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def sinkhorn_matrix(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    iterations: int = 20,
) -> Tensor:
    """Return a differentiable approximately doubly-stochastic matrix."""

    if logits.shape[-1] != logits.shape[-2]:
        raise ValueError("Sinkhorn logits must be square")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    log_p = logits / temperature
    for _ in range(iterations):
        log_p = log_p - torch.logsumexp(
            log_p, dim=-1, keepdim=True
        )
        log_p = log_p - torch.logsumexp(
            log_p, dim=-2, keepdim=True
        )
    return torch.exp(log_p)


@torch.no_grad()
def maximum_weight_permutation(
    scores: Tensor,
    *,
    max_modules: int = 12,
) -> Tensor:
    """Exact maximum-weight assignment by bitmask dynamic programming.

    This projection is intended for controller execution/evaluation at modest
    module count.  Complexity is O(M^2 2^M); it is not the final large-M router.
    """

    if scores.shape[-1] != scores.shape[-2]:
        raise ValueError("scores must be square")
    modules = scores.shape[-1]
    if modules > max_modules:
        raise ValueError(
            f"exact assignment supports at most {max_modules} modules; got {modules}"
        )

    leading = scores.shape[:-2]
    flat = scores.detach().float().cpu().reshape(-1, modules, modules)
    results: list[list[int]] = []

    for matrix in flat:
        # mask -> (score, path) after assigning the first popcount(mask) rows.
        dp: dict[int, tuple[float, list[int]]] = {0: (0.0, [])}
        for row in range(modules):
            nxt: dict[int, tuple[float, list[int]]] = {}
            for mask, (value, path) in dp.items():
                for col in range(modules):
                    bit = 1 << col
                    if mask & bit:
                        continue
                    new_mask = mask | bit
                    new_value = value + float(matrix[row, col])
                    old = nxt.get(new_mask)
                    if old is None or new_value > old[0]:
                        nxt[new_mask] = (
                            new_value,
                            path + [col],
                        )
            dp = nxt
        full_mask = (1 << modules) - 1
        results.append(dp[full_mask][1])

    output = torch.tensor(
        results,
        dtype=torch.long,
        device=scores.device,
    )
    return output.reshape(leading + (modules,))


def permutation_matrix_batch(
    permutation: Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Convert gather permutations [..., M] to [..., M, M] matrices."""

    if permutation.dtype != torch.long:
        raise ValueError("permutation must use torch.long")
    modules = permutation.shape[-1]
    return torch.nn.functional.one_hot(
        permutation, num_classes=modules
    ).to(dtype=dtype)


class TaskPermutationController(nn.Module):
    """Small research controller mapping task id to a permutation matrix.

    Training uses a Sinkhorn relaxation.  Execution projects the soft matrix to
    an exact one-to-one assignment, which can then drive the monomial scan.
    """

    def __init__(
        self,
        num_tasks: int,
        num_modules: int,
        *,
        sinkhorn_iterations: int = 20,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if num_tasks <= 0 or num_modules <= 1:
            raise ValueError("num_tasks must be positive and num_modules > 1")
        self.num_tasks = num_tasks
        self.num_modules = num_modules
        self.sinkhorn_iterations = sinkhorn_iterations
        logits = init_scale * torch.randn(
            num_tasks, num_modules, num_modules
        )
        self.logits = nn.Parameter(logits)

    def soft_matrix(
        self,
        task_id: Tensor,
        *,
        temperature: float = 1.0,
    ) -> Tensor:
        return sinkhorn_matrix(
            self.logits[task_id],
            temperature=temperature,
            iterations=self.sinkhorn_iterations,
        )

    @torch.no_grad()
    def hard_permutation(
        self,
        task_id: Tensor,
        *,
        temperature: float = 0.1,
    ) -> Tensor:
        soft = self.soft_matrix(
            task_id, temperature=temperature
        )
        return maximum_weight_permutation(soft)

    @torch.no_grad()
    def hard_matrix(
        self,
        task_id: Tensor,
        *,
        temperature: float = 0.1,
    ) -> Tensor:
        permutation = self.hard_permutation(
            task_id, temperature=temperature
        )
        return permutation_matrix_batch(
            permutation, dtype=self.logits.dtype
        )


def geometric_temperature(
    step: int,
    total_steps: int,
    *,
    start: float,
    end: float,
) -> float:
    """Geometric annealing schedule for positive temperatures."""

    if total_steps <= 1:
        return end
    if start <= 0 or end <= 0:
        raise ValueError("temperatures must be positive")
    fraction = min(max(step / (total_steps - 1), 0.0), 1.0)
    return start * math.exp(
        math.log(end / start) * fraction
    )


class FactorizedTaskPermutationController(nn.Module):
    """Low-bandwidth task-row query / shared-source-key permutation controller.

    Scores are factorized as

        score[t, i, j] = <query[t, i], key[j]> / sqrt(rank)

    so parameter count is M * rank * (T + 1), versus T * M^2 for the full
    TaskPermutationController.
    """

    def __init__(
        self,
        num_tasks: int,
        num_modules: int,
        rank: int,
        *,
        sinkhorn_iterations: int = 20,
        init_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if num_tasks <= 0 or num_modules <= 1 or rank <= 0:
            raise ValueError(
                "num_tasks/rank must be positive and num_modules > 1"
            )
        self.num_tasks = num_tasks
        self.num_modules = num_modules
        self.rank = rank
        self.sinkhorn_iterations = sinkhorn_iterations
        self.query = nn.Parameter(
            init_scale * torch.randn(num_tasks, num_modules, rank)
        )
        self.key = nn.Parameter(
            init_scale * torch.randn(num_modules, rank)
        )

    def score_matrix(self, task_id: Tensor) -> Tensor:
        query = self.query[task_id]
        return torch.einsum(
            "...ir,jr->...ij", query, self.key
        ) / math.sqrt(self.rank)

    def soft_matrix(
        self,
        task_id: Tensor,
        *,
        temperature: float = 1.0,
    ) -> Tensor:
        return sinkhorn_matrix(
            self.score_matrix(task_id),
            temperature=temperature,
            iterations=self.sinkhorn_iterations,
        )

    @torch.no_grad()
    def hard_permutation(
        self,
        task_id: Tensor,
        *,
        temperature: float = 0.1,
    ) -> Tensor:
        soft = self.soft_matrix(
            task_id, temperature=temperature
        )
        return maximum_weight_permutation(soft)

    @torch.no_grad()
    def hard_matrix(
        self,
        task_id: Tensor,
        *,
        temperature: float = 0.1,
    ) -> Tensor:
        permutation = self.hard_permutation(
            task_id, temperature=temperature
        )
        return permutation_matrix_batch(
            permutation, dtype=self.query.dtype
        )


class MatchingPermutationController(nn.Module):
    """Shared content/context-conditioned permutation controller.

    Source and target routing descriptors are projected into a common embedding
    space.  Pairwise scores are normalized dot products, followed by Sinkhorn
    during training and exact one-to-one assignment during hard execution.

    No task-ID table is used.
    """

    def __init__(
        self,
        context_dim: int,
        rank: int,
        *,
        sinkhorn_iterations: int = 20,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if context_dim <= 0 or rank <= 0:
            raise ValueError("context_dim and rank must be positive")
        self.context_dim = context_dim
        self.rank = rank
        self.sinkhorn_iterations = sinkhorn_iterations
        self.normalize = normalize
        self.source_projection = nn.Linear(
            context_dim, rank, bias=False
        )
        self.target_projection = nn.Linear(
            context_dim, rank, bias=False
        )

    def score_matrix(
        self,
        source_context: Tensor,
        target_context: Tensor,
    ) -> Tensor:
        if source_context.shape != target_context.shape:
            raise ValueError(
                "source_context and target_context must have identical shapes"
            )
        if source_context.shape[-1] != self.context_dim:
            raise ValueError("unexpected context dimension")

        source = self.source_projection(source_context)
        target = self.target_projection(target_context)
        if self.normalize:
            source = torch.nn.functional.normalize(
                source, dim=-1, eps=1e-8
            )
            target = torch.nn.functional.normalize(
                target, dim=-1, eps=1e-8
            )
            scale = 1.0
        else:
            scale = math.sqrt(self.rank)
        return torch.matmul(
            target, source.transpose(-1, -2)
        ) / scale

    def soft_matrix(
        self,
        source_context: Tensor,
        target_context: Tensor,
        *,
        temperature: float = 1.0,
    ) -> Tensor:
        return sinkhorn_matrix(
            self.score_matrix(source_context, target_context),
            temperature=temperature,
            iterations=self.sinkhorn_iterations,
        )

    @torch.no_grad()
    def hard_permutation(
        self,
        source_context: Tensor,
        target_context: Tensor,
        *,
        temperature: float = 0.1,
    ) -> Tensor:
        soft = self.soft_matrix(
            source_context,
            target_context,
            temperature=temperature,
        )
        return maximum_weight_permutation(soft)
