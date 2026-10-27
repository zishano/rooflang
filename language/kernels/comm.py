# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Communication Kernel subclasses for collective operations.

These model the cost of NCCL-style collectives at sharding boundaries.

All collectives use the ring algorithm model:
  transferred_bytes = coeff × (W-1)/W × total_bytes

where coeff=1 for single-phase (Scatter, Gather, Broadcast, Reduce,
ReduceScatter, AllGather) and coeff=2 for two-phase (AllReduce = RS + AG).

total_bytes is the logical tensor size involved in the collective.

Reduce-included collectives (AllReduce, ReduceScatter, Reduce) perform
actual arithmetic (element-wise addition across ranks), so flops > 0:
  flops = (W-1)/W × n_elements
"""

from rooflang.language.kernels.kernel import Kernel
from rooflang.language.utils import dtype_bytes


class CommKernel(Kernel):
    """Base class for all communication kernels.

    Communication kernels do not require placement — their cost is
    derived from neighboring placed kernels in the simulator.

    Port constraints (enforced by validate_ports):
      Scatter/Broadcast: 1 input, world outputs
      Gather/Reduce:     world inputs, 1 output
      AllReduce/AllGather/ReduceScatter/AllToAll: world inputs, world outputs
    """
    _requires_placement = False
    _single_input = False   # True → exactly 1 input port
    _single_output = False  # True → exactly 1 output port

    def validate_ports(self) -> None:
        """Check port counts match the single-tensor constraint."""
        if not hasattr(self, 'world'):
            return
        if self._single_input and len(self.inputs) != 1:
            raise ValueError(
                f"{type(self).__name__}: expected 1 input, got {len(self.inputs)}")
        if not self._single_input and len(self.inputs) != self.world:
            raise ValueError(
                f"{type(self).__name__}: expected {self.world} inputs, "
                f"got {len(self.inputs)}")
        if self._single_output and len(self.outputs) != 1:
            raise ValueError(
                f"{type(self).__name__}: expected 1 output, got {len(self.outputs)}")
        if not self._single_output and len(self.outputs) != self.world:
            raise ValueError(
                f"{type(self).__name__}: expected {self.world} outputs, "
                f"got {len(self.outputs)}")


class AllReduce(CommKernel):
    """All-reduce: every rank ends with the full reduced result.

    Two-phase (ReduceScatter + AllGather):
      transferred_bytes = 2 × (W-1)/W × total_bytes.
    """
    _single_input = False
    _single_output = False

    def __init__(self, total_bytes: float, world: int, dtype: str = "bf16"):
        self.total_bytes = total_bytes
        self.world = world
        self.dtype_ = dtype
        super().__init__()

    @property
    def flops(self) -> float:
        n_elements = self.total_bytes / dtype_bytes(self.dtype_)
        return (self.world - 1) / self.world * n_elements

    @property
    def transferred_bytes(self) -> float:
        return 2.0 * (self.world - 1) / self.world * self.total_bytes

    @property
    def input_bytes(self) -> float:
        return self.total_bytes

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.total_bytes


class ReduceScatter(CommKernel):
    """Reduce-scatter: each rank gets 1/W of the reduced result.

    transferred_bytes = (W-1)/W × total_bytes.
    """
    _single_input = False
    _single_output = False

    def __init__(self, total_bytes: float, world: int, dtype: str = "bf16"):
        self.total_bytes = total_bytes
        self.world = world
        self.dtype_ = dtype
        super().__init__()

    @property
    def flops(self) -> float:
        n_elements = self.total_bytes / dtype_bytes(self.dtype_)
        return (self.world - 1) / self.world * n_elements

    @property
    def transferred_bytes(self) -> float:
        return (self.world - 1) / self.world * self.total_bytes

    @property
    def input_bytes(self) -> float:
        return self.total_bytes

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.total_bytes / self.world


class AllGather(CommKernel):
    """All-gather: each rank broadcasts its shard; all get the full tensor.

    transferred_bytes = (W-1)/W × total_bytes.
    """
    _single_input = False
    _single_output = False

    def __init__(self, total_bytes: float, world: int):
        self.total_bytes = total_bytes
        self.world = world
        super().__init__()

    @property
    def transferred_bytes(self) -> float:
        return (self.world - 1) / self.world * self.total_bytes

    @property
    def input_bytes(self) -> float:
        return self.total_bytes / self.world

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.total_bytes


class AllToAll(CommKernel):
    """All-to-all: each rank sends a distinct chunk to every other rank.

    transferred_bytes = (W-1)/W × total_bytes.
    """
    _single_input = False
    _single_output = False

    def __init__(self, total_bytes: float, world: int):
        self.total_bytes = total_bytes
        self.world = world
        super().__init__()

    @property
    def transferred_bytes(self) -> float:
        return (self.world - 1) / self.world * self.total_bytes

    @property
    def input_bytes(self) -> float:
        return self.total_bytes

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.total_bytes


class Broadcast(CommKernel):
    """Broadcast: root sends full payload to all ranks.

    Bottleneck link (root's outgoing) carries the full tensor.
    transferred_bytes = total_bytes.
    """
    _single_input = True
    _single_output = False

    def __init__(self, total_bytes: float, world: int):
        self.total_bytes = total_bytes
        self.world = world
        super().__init__()

    @property
    def transferred_bytes(self) -> float:
        return self.total_bytes

    @property
    def input_bytes(self) -> float:
        return self.total_bytes

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.total_bytes


class Scatter(CommKernel):
    """Scatter: root distributes distinct slices to each rank.

    transferred_bytes = (W-1)/W × total_bytes.
    """
    _single_input = True
    _single_output = False

    def __init__(self, total_bytes: float, world: int, dim: int = 0):
        self.total_bytes = total_bytes
        self.world = world
        self.dim = dim
        super().__init__()

    @property
    def transferred_bytes(self) -> float:
        return (self.world - 1) / self.world * self.total_bytes

    @property
    def input_bytes(self) -> float:
        return self.total_bytes

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.total_bytes / self.world


class CyclicScatter(Scatter):
    """Scatter sequence rows by cyclic owner instead of contiguous chunks.

    For interleave size one, rank ``r`` receives rows ``r, r+world, ...``.
    The network volume is the same as Scatter; this subtype preserves the
    placement rule so traces and tests can distinguish the data layout.
    """

    def __init__(self, total_bytes: float, world: int, dim: int = 0,
                 interleave_size: int = 1):
        if interleave_size <= 0:
            raise ValueError("interleave_size must be positive")
        self.interleave_size = interleave_size
        self.shard_scheme = "cyclic"
        super().__init__(total_bytes=total_bytes, world=world, dim=dim)


class Gather(CommKernel):
    """Gather: each rank sends its slice to root which concatenates.

    transferred_bytes = (W-1)/W × total_bytes.
    """
    _single_input = False
    _single_output = True

    def __init__(self, total_bytes: float, world: int, dim: int = 0):
        self.total_bytes = total_bytes
        self.world = world
        self.dim = dim
        super().__init__()

    @property
    def transferred_bytes(self) -> float:
        return (self.world - 1) / self.world * self.total_bytes

    @property
    def input_bytes(self) -> float:
        return self.total_bytes / self.world

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.total_bytes


class Reduce(CommKernel):
    """Reduce: each rank sends full buffer; root reduces element-wise.

    Bottleneck link (into root) carries the full tensor.
    transferred_bytes = total_bytes.
    """
    _single_input = False
    _single_output = True

    def __init__(self, total_bytes: float, world: int, dtype: str = "bf16"):
        self.total_bytes = total_bytes
        self.world = world
        self.dtype_ = dtype
        super().__init__()

    @property
    def flops(self) -> float:
        n_elements = self.total_bytes / dtype_bytes(self.dtype_)
        return (self.world - 1) / self.world * n_elements

    @property
    def transferred_bytes(self) -> float:
        return self.total_bytes

    @property
    def input_bytes(self) -> float:
        return self.total_bytes

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.total_bytes


class AttentionLseReduce(Reduce):
    """Merge normalized attention partials with log-sum-exp statistics.

    This is not an element-wise ``Reduce``.  Every rank sends its locally
    normalized output together with ``(max_logit, exp_sum)`` statistics.  The
    merged output is

        sum_i exp(local_max_i - global_max) * local_sum_i * local_output_i
        ------------------------------------------------------------------
                    sum_i exp(local_max_i - global_max) * local_sum_i

    The reduced ports expose both payloads so placement and simulator memory
    accounting see the statistic traffic.  ``total_bytes`` remains the logical
    output size for compatibility with communication cost conventions.
    """

    _single_input = True
    _single_output = True

    def __init__(
        self, total_bytes: float, world: int, dtype: str = "bf16",
        stats_bytes_per_rank: float = 0.0,
    ):
        if stats_bytes_per_rank < 0:
            raise ValueError("stats_bytes_per_rank must be non-negative")
        self.stats_bytes_per_rank = stats_bytes_per_rank
        self.split_extra_inputs = {"attn_stats": "s{i}"}
        super().__init__(total_bytes=total_bytes, world=world, dtype=dtype)

    def validate_ports(self) -> None:
        if len(self.inputs) != 2 * self.world or len(self.outputs) != 1:
            raise ValueError(
                "AttentionLseReduce: expected world output partials, world "
                "statistic tensors, and one merged output")

    @property
    def flops(self) -> float:
        # Weighted output accumulation, stable global max, and denominator.
        n_elements = self.total_bytes / dtype_bytes(self.dtype_)
        return float((self.world - 1) * n_elements * 3 +
                     self.world * 4)

    @property
    def input_bytes(self) -> float:
        return self.total_bytes + self.world * self.stats_bytes_per_rank

    @property
    def transferred_bytes(self) -> float:
        return self.total_bytes + self.world * self.stats_bytes_per_rank


class SparseGlobalSelect(Scatter):
    """Scatter caches while preserving exact global sparse selection.

    A plain context ``Scatter`` lets each rank select a local top-k.  This
    fusion models the required preparation instead: local index scores are
    exchanged to select one global top-k set, then the selected KV entries are
    routed to the ranks that consume them.  The index shard remains the
    visible tensor payload; selection metadata and selected-KV traffic are
    explicit attributes rather than unaccounted implementation details.
    """

    def __init__(
        self, total_bytes: float, world: int, dim: int = 0, *,
        selected_kv_bytes: float = 0.0,
        selection_metadata_bytes: float = 0.0,
        selection_flops: float = 0.0,
    ):
        if min(selected_kv_bytes, selection_metadata_bytes,
               selection_flops) < 0:
            raise ValueError("sparse selection costs must be non-negative")
        self.selected_kv_bytes = selected_kv_bytes
        self.selection_metadata_bytes = selection_metadata_bytes
        self.selection_flops = selection_flops
        super().__init__(total_bytes=total_bytes, world=world, dim=dim)

    @property
    def flops(self) -> float:
        return self.selection_flops

    @property
    def transferred_bytes(self) -> float:
        remote_fraction = (self.world - 1) / self.world
        return (
            super().transferred_bytes
            + remote_fraction
            * (self.selected_kv_bytes + self.selection_metadata_bytes)
        )


class Send(CommKernel):
    """Point-to-point send (e.g. PP stage boundary, sender side).

    transferred_bytes = total_bytes.
    """

    def __init__(self, total_bytes: float):
        self.total_bytes = total_bytes
        super().__init__()

    @property
    def transferred_bytes(self) -> float:
        return self.total_bytes

    @property
    def input_bytes(self) -> float:
        return self.total_bytes

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return 0.0


class Recv(CommKernel):
    """Point-to-point recv (e.g. PP stage boundary, receiver side).

    transferred_bytes = total_bytes.
    """

    def __init__(self, total_bytes: float):
        self.total_bytes = total_bytes
        super().__init__()

    @property
    def transferred_bytes(self) -> float:
        return self.total_bytes

    @property
    def input_bytes(self) -> float:
        return 0.0

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.total_bytes
