"""Explicit state and attention operators for the DSV4 DCP graph.

Window rows use fixed ring slots; compressed rows follow the window. Cyclic
ownership refers to these physical slots, so a sliding window does not move
the entire cache. Numerical methods operate on already encoded Q/KV/Index-K.
The learned projection/compression and FP4/FP8 quantizers remain RoofLang
cost abstractions, as in the original model.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from rooflang.language.kernels.kernel import Kernel
from rooflang.language.kernels.comm import AttentionLseReduce
from rooflang.language.tensor import Tensor


@dataclass(frozen=True)
class DCPStateSpec:
    context: int
    window: int
    ratio: int
    compressed_capacity: int
    cp: int
    rank: int
    batch: int
    kv_dim: int
    index_dim: int
    heads: int
    head_dim: int
    topk: int

    @property
    def capacity(self):
        return self.window + self.compressed_capacity

    @property
    def local_capacity(self):
        return (self.capacity + self.cp - 1 - self.rank) // self.cp

    @property
    def compressed(self):
        return (self.context + 1) // self.ratio if self.ratio else 0

    @property
    def emits_compressed(self):
        return bool(self.ratio and (self.context + 1) % self.ratio == 0)

    @property
    def window_slots(self):
        return tuple(position % self.window for position in range(
            max(0, self.context + 1 - self.window), self.context + 1))

    @property
    def selected_compressed(self):
        return min(self.topk, self.compressed) if self.ratio == 4 else self.compressed


class DCPStateLoad(Kernel):
    """Materialize a rank-local persistent allocation from preload buffers."""

    def __init__(self, tensor):
        super().__init__(inputs={"x": Tensor(tensor.dtype, tensor.shape)},
                         outputs={"y": Tensor(tensor.dtype, tensor.shape)})

    def evaluate(self, inputs):
        return {"y": np.array(inputs["x"], copy=True)}


class DCPKVUpdate(Kernel):
    """Write the new window row and an optional compressed row in place."""

    def __init__(self, spec):
        self.spec = spec
        self.writes = {"new_kv": spec.context % spec.window}
        if spec.emits_compressed:
            self.writes["compressed_kv"] = spec.window + spec.compressed - 1
        shape = (spec.batch, spec.local_capacity, spec.kv_dim)
        inputs = {"cache": Tensor("fp8", shape)}
        inputs.update({port: Tensor("fp8", (spec.batch, 1, spec.kv_dim))
                       for port in self.writes})
        super().__init__(inputs=inputs, outputs={"updated": Tensor("fp8", shape)})
        self.inplace_aliases = {"updated": "cache"}

    def input_read_fraction(self, port):
        if port == "cache":
            return 0.0
        return float(self.writes[port] % self.spec.cp == self.spec.rank)

    def output_write_fraction(self, port):
        count = sum(slot % self.spec.cp == self.spec.rank
                    for slot in self.writes.values())
        return Fraction(count, self.spec.local_capacity)

    def evaluate(self, inputs):
        result = np.array(inputs["cache"], copy=True)
        for port, slot in self.writes.items():
            if slot % self.spec.cp == self.spec.rank:
                result[:, slot // self.spec.cp] = inputs[port][:, 0]
        return {"updated": result}


class DCPIndexUpdate(Kernel):
    """Maintain one explicit full Index-K replica on this CP rank."""

    def __init__(self, spec):
        self.spec = spec
        shape = (spec.batch, spec.compressed_capacity, spec.index_dim)
        inputs = {"cache": Tensor("fp4", shape)}
        if spec.emits_compressed:
            inputs["new_index"] = Tensor("fp4", (spec.batch, 1, spec.index_dim))
        super().__init__(inputs=inputs, outputs={"updated": Tensor("fp4", shape)})
        self.inplace_aliases = {"updated": "cache"}

    def input_read_fraction(self, port):
        return 0.0 if port == "cache" else 1.0

    def output_write_fraction(self, port):
        return Fraction(int(self.spec.emits_compressed), self.spec.compressed_capacity)

    def evaluate(self, inputs):
        result = np.array(inputs["cache"], copy=True)
        if self.spec.emits_compressed:
            result[:, self.spec.compressed - 1] = inputs["new_index"][:, 0]
        return {"updated": result}


class DCPGlobalTopK(Kernel):
    """Full replicated Index-K scoring and absolute compressed-row top-k.

    The index query is encoded in the first index_dim features of each
    attention-query head for the numerical reference. Production indexer
    FLOPs retain the original fused Indexer score/reduction cost model.
    """

    def __init__(self, spec, index_heads):
        self.spec, self.index_heads = spec, index_heads
        super().__init__(inputs={
            "q": Tensor("bf16", (spec.batch, 1, spec.heads * spec.head_dim)),
            "index_kv": Tensor("fp4", (spec.batch, spec.compressed_capacity,
                                        spec.index_dim)),
        }, outputs={"ids": Tensor("int64", (spec.batch, spec.selected_compressed))})

    @property
    def flops(self):
        s = self.spec
        return float(s.batch * self.index_heads * s.compressed * (2 * s.index_dim + 3))

    @property
    def flops_by_dtype(self):
        return {"fp4": self.flops} if self.flops else {}

    def input_read_fraction(self, port):
        return (Fraction(self.spec.compressed, self.spec.compressed_capacity)
                if port == "index_kv" else 1.0)

    def evaluate(self, inputs):
        s = self.spec
        q = inputs["q"].reshape(s.batch, s.heads, s.head_dim)
        iq = q[:, :self.index_heads, :s.index_dim]
        cache = inputs["index_kv"][:, :s.compressed]
        scores = np.maximum(np.einsum("bhi,bni->bhn", iq, cache), 0).sum(axis=1)
        positions = np.arange(s.compressed)
        ids = np.stack([np.lexsort((positions, -row))[:s.selected_compressed]
                        for row in scores])
        return {"ids": ids.astype(np.int64)}


class DCPOwnerLocalRows(Kernel):
    """Filter global selections by cyclic owner and read those local rows.

    Allocation allows the maximum possible local top-k skew. The timing
    model uses the expected uniform owner count, explicitly recorded here.
    """

    def __init__(self, spec):
        self.spec = spec
        local_window = sum(slot % spec.cp == spec.rank for slot in spec.window_slots)
        local_compressed = sum((spec.window + i) % spec.cp == spec.rank
                               for i in range(spec.compressed))
        self.max_rows = local_window + min(spec.selected_compressed, local_compressed)
        self.expected_rows = Fraction(local_window)
        if spec.compressed:
            self.expected_rows += Fraction(spec.selected_compressed * local_compressed,
                                           spec.compressed)
        # A zero-length local shard is valid; masks represent empty owners.
        inputs = {"cache": Tensor("fp8", (spec.batch, spec.local_capacity, spec.kv_dim))}
        if spec.ratio == 4:
            inputs["ids"] = Tensor("int64", (spec.batch, spec.selected_compressed))
        super().__init__(inputs=inputs, outputs={
            "kv": Tensor("fp8", (spec.batch, self.max_rows, spec.kv_dim)),
            "mask": Tensor("int32", (spec.batch, self.max_rows)),
        })

    @property
    def flops(self):
        return float(2 * self.spec.batch * self.spec.selected_compressed)

    def input_read_fraction(self, port):
        return (self.expected_rows / self.spec.local_capacity
                if port == "cache" else 1.0)

    def output_write_fraction(self, port):
        return (self.expected_rows / self.max_rows
                if port == "kv" and self.max_rows else 1.0)

    def evaluate(self, inputs):
        s = self.spec
        values = np.zeros((s.batch, self.max_rows, s.kv_dim))
        mask = np.zeros((s.batch, self.max_rows), dtype=np.int32)
        for batch in range(s.batch):
            compressed = inputs["ids"][batch] if s.ratio == 4 else range(s.compressed)
            slots = [*s.window_slots, *(s.window + int(i) for i in compressed)]
            local = [slot // s.cp for slot in slots if slot % s.cp == s.rank]
            if local:
                values[batch, :len(local)] = inputs["cache"][batch, local]
                mask[batch, :len(local)] = 1
        return {"kv": values, "mask": mask}


class DCPAttentionPartial(Kernel):
    """Per-head local softmax output plus (max, exp_sum) LSE statistics."""

    def __init__(self, spec, rows):
        self.spec, self.expected_rows = spec, rows.expected_rows
        self.max_rows = rows.max_rows
        self.B, self.H, self.S_q, self.Hd = spec.batch, spec.heads, 1, spec.head_dim
        self.cp_rank, self.cp_world = spec.rank, spec.cp
        self.global_selection = True
        super().__init__(inputs={
            "q": Tensor("bf16", (spec.batch, 1, spec.heads * spec.head_dim)),
            "kv": Tensor("fp8", rows.outputs["kv"].shape),
            "mask": Tensor("int32", rows.outputs["mask"].shape),
        }, outputs={
            "y": Tensor("bf16", (spec.batch, 1, spec.heads * spec.head_dim)),
            "attn_stats": Tensor("fp32", (spec.batch, spec.heads, 1, 2)),
        })

    @property
    def flops(self):
        s = self.spec
        return float(4 * s.batch * s.heads * s.head_dim * self.expected_rows)

    @property
    def flops_by_dtype(self):
        return {"fp8": self.flops} if self.flops else {}

    def input_read_fraction(self, port):
        return (self.expected_rows / self.max_rows
                if port == "kv" and self.max_rows else 1.0)

    def evaluate(self, inputs):
        s = self.spec
        q = inputs["q"].reshape(s.batch, s.heads, s.head_dim)
        output = np.zeros_like(q, dtype=np.float64)
        stats = np.zeros((s.batch, s.heads, 1, 2))
        stats[..., 0] = -np.inf
        for batch in range(s.batch):
            kv = inputs["kv"][batch, inputs["mask"][batch].astype(bool)]
            if not len(kv):
                continue
            logits = q[batch] @ kv.T
            maxima = logits.max(axis=-1)
            weights = np.exp(logits - maxima[:, None])
            sums = weights.sum(axis=-1)
            output[batch] = (weights @ kv) / sums[:, None]
            stats[batch, :, 0, 0] = maxima
            stats[batch, :, 0, 1] = sums
        return {"y": output.reshape(s.batch, 1, s.heads * s.head_dim),
                "attn_stats": stats}


class DCPAttentionMerge(AttentionLseReduce):
    def __init__(self, spec):
        self.spec = spec
        y_shape = (spec.batch, 1, spec.heads * spec.head_dim)
        stats_shape = (spec.batch, spec.heads, 1, 2)
        super().__init__(spec.batch * spec.heads * spec.head_dim * 2,
                         spec.cp, stats_bytes_per_rank=spec.batch * spec.heads * 8)
        self.inputs = {**{f"i{r}": Tensor("bf16", y_shape) for r in range(spec.cp)},
                       **{f"s{r}": Tensor("fp32", stats_shape) for r in range(spec.cp)}}
        self.outputs = {"y": Tensor("bf16", y_shape)}

    @property
    def flops(self):
        s = self.spec
        elements = s.batch * s.heads * s.head_dim
        return float((s.cp - 1) * elements * 3 + s.cp * s.batch * s.heads * 4)

    def evaluate(self, inputs):
        s = self.spec
        outputs = np.stack([inputs[f"i{r}"].reshape(s.batch, s.heads, s.head_dim)
                            for r in range(s.cp)])
        stats = np.stack([inputs[f"s{r}"][:, :, 0] for r in range(s.cp)])
        maxima, sums = stats[..., 0], stats[..., 1]
        global_max = maxima.max(axis=0)
        weights = np.exp(maxima - global_max[None]) * sums
        result = (outputs * weights[..., None]).sum(axis=0) / weights.sum(axis=0)[..., None]
        return {"y": result.reshape(s.batch, 1, s.heads * s.head_dim)}
