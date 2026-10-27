# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Utilities for decode context-parallel modeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GRAPH_STRATEGIES = {
    "corrected-default": {
        "name": "corrected-default",
        "pass_order": (
            "dp", "dp-comm-normalization",
            "exact_sparse_selection", "lse_merge", "comm_fusion",
            "placement",
        ),
        "comm_fusion": True,
        "layout": "uniform-context-v1",
        "placement": "stage-gpu-order",
        "schedule": "simulator-single-stream",
    },
    "dual-stream-shared-expert": {
        "name": "dual-stream-shared-expert",
        "pass_order": (
            "dp", "dp-comm-normalization",
            "exact_sparse_selection", "lse_merge", "comm_fusion",
            "dual-stream-schedule", "placement",
        ),
        "comm_fusion": True,
        "layout": "uniform-context-v1",
        "placement": "stage-gpu-order",
        "schedule": "dual-stream-shared-expert",
        "shared_expert_stream": 1,
    },
}


def resolve_graph_strategy(strategy=None) -> dict:
    """Validate and materialize a serializable graph strategy."""
    if strategy is None:
        strategy = "corrected-default"
    if isinstance(strategy, str):
        try:
            return dict(GRAPH_STRATEGIES[strategy])
        except KeyError as error:
            raise ValueError(
                f"unknown graph strategy {strategy!r}") from error
    if not isinstance(strategy, dict):
        raise TypeError("graph_strategy must be a name or dictionary")
    value = dict(strategy)
    if value.get("layout", "uniform-context-v1") \
            != "uniform-context-v1":
        raise ValueError("unsupported graph layout")
    if not isinstance(value.get("comm_fusion", True), bool):
        raise ValueError("graph_strategy.comm_fusion must be boolean")
    value.setdefault("name", "custom")
    value.setdefault(
        "pass_order",
        (
            "dp", "dp-comm-normalization",
            "exact_sparse_selection", "lse_merge",
            "comm_fusion" if value.get("comm_fusion", True) else "no-op",
            "placement",
        ),
    )
    value.setdefault("placement", "stage-gpu-order")
    value.setdefault("schedule", "simulator-single-stream")
    return value


def stable_attention_merge(local_output, local_max, local_sum):
    """Numerically stable merge of normalized attention shard partials.

    The last dimension is the attention value dimension.  ``local_max`` and
    ``local_sum`` have shape ``(world, ..., 1)`` and ``local_output`` has shape
    ``(world, ..., value_dim)``.
    """
    output = np.asarray(local_output, dtype=np.float64)
    maxima = np.asarray(local_max, dtype=np.float64)
    sums = np.asarray(local_sum, dtype=np.float64)
    if output.ndim != maxima.ndim \
            or output.shape[:-1] != maxima.shape[:-1] \
            or maxima.shape[-1] != 1:
        raise ValueError("local attention and statistic shapes disagree")
    if maxima.shape != sums.shape:
        raise ValueError("local max and sum shapes disagree")
    global_max = np.max(maxima, axis=0, keepdims=True)
    scaled = np.exp(maxima - global_max)
    weights = scaled * sums
    denominator = np.sum(weights, axis=0, keepdims=True)
    if np.any(denominator == 0):
        raise ValueError("attention denominator is zero")
    numerator = np.sum(weights * output, axis=0)
    return numerator / np.squeeze(denominator, axis=0)


def exact_global_topk(score_shards, index_shards, k: int):
    """Select one global top-k set from disjoint score shards.

    ``index_shards`` contain cache-absolute ids, not rank-local ids.  Ties are
    broken by the cache id so distributed and unsplit execution are stable.
    """
    if len(score_shards) != len(index_shards):
        raise ValueError("score and index shard counts differ")
    scores = np.concatenate(
        [np.asarray(value).reshape(-1) for value in score_shards])
    indices = np.concatenate(
        [np.asarray(value, dtype=np.int64).reshape(-1)
         for value in index_shards])
    if scores.size != indices.size:
        raise ValueError("score and index element counts differ")
    if not 0 <= k <= scores.size:
        raise ValueError(f"top-k={k} outside [0, {scores.size}]")
    order = np.lexsort((indices, -scores))[:k]
    return scores[order], indices[order]


@dataclass(frozen=True)
class KVLayout:
    """Auditable physical KV layout for one layer and cache kind."""

    layer: int
    cache: str
    dtype: str
    request_start: int
    request_stop: int
    cache_index_start: int
    cache_index_stop: int
    owner_gpus: tuple[str, ...]
    replicas: int
    bytes_per_replica: int
    token_map: tuple[tuple[int, int, int, int], ...]
    shard_scheme: str = "contiguous"
    rank: int = 0
    shard_world: int = 1
    replicated: bool = False
    lifecycle: str = "preload-to-stage-output"

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "cache": self.cache,
            "dtype": self.dtype,
            "request_range": [
                self.request_start, self.request_stop],
            "cache_index_range": [
                self.cache_index_start, self.cache_index_stop],
            "owner_gpus": list(self.owner_gpus),
            "replicas": self.replicas,
            "bytes_per_replica": self.bytes_per_replica,
            "token_map": [list(item) for item in self.token_map],
            "shard_scheme": self.shard_scheme,
            "rank": self.rank,
            "shard_world": self.shard_world,
            "replicated": self.replicated,
            "lifecycle": self.lifecycle,
        }


def owner_localize_positions(positions, cp: int, rank: int,
                             interleave_size: int = 1):
    """Map global cache positions to owner-local positions.

    With the default interleave of one, rank ``r`` owns positions
    ``r, r + cp, r + 2*cp, ...``.  The returned pairs preserve input order.
    """
    if cp <= 0 or not 0 <= rank < cp or interleave_size <= 0:
        raise ValueError("invalid CP rank or interleave size")
    result = []
    for position in positions:
        position = int(position)
        if position < 0:
            raise ValueError("cache positions must be non-negative")
        owner = (position // interleave_size) % cp
        if owner == rank:
            local = (position // (interleave_size * cp)) * interleave_size
            local += position % interleave_size
            result.append((position, local))
    return result


class DCPKVCache:
    """Small numerical oracle for persistent, cyclically sharded KV state.

    Values are keyed by request and global cache position.  Appends mutate
    the persistent logical cache; each step's read reconstructs the ordered
    history from owner-local shards.  This is an oracle utility, not a GPU
    cache implementation.
    """

    def __init__(self, cp: int, initial_values, interleave_size: int = 1,
                 index_values=None):
        if cp <= 0 or interleave_size <= 0:
            raise ValueError("cp and interleave_size must be positive")
        import numpy as np

        values = np.asarray(initial_values)
        if values.ndim < 2:
            raise ValueError("initial_values must have request and token axes")
        indices = values if index_values is None else np.asarray(index_values)
        if indices.ndim < 2 or indices.shape[:2] != values.shape[:2]:
            raise ValueError(
                "index_values must match request and token dimensions")
        self.cp = cp
        self.interleave_size = interleave_size
        self._value_shape = tuple(values.shape[2:])
        self._index_shape = tuple(indices.shape[2:])
        self._lengths = [int(values.shape[1]) for _ in range(values.shape[0])]
        self._shards = [dict() for _ in range(cp)]
        self._index_shards = [dict() for _ in range(cp)]
        for request in range(values.shape[0]):
            for position in range(values.shape[1]):
                owner, local = self.owner_local(request, position)
                self._shards[owner][(request, local)] = values[
                    request, position].copy()
                self._index_shards[owner][(request, local)] = indices[
                    request, position].copy()

    def owner_local(self, request: int, position: int) -> tuple[int, int]:
        if request < 0 or request >= len(self._lengths):
            raise IndexError("request id is outside the cache")
        if position < 0:
            raise ValueError("cache positions must be non-negative")
        owner = (position // self.interleave_size) % self.cp
        local = (position // (self.interleave_size * self.cp)) \
            * self.interleave_size + position % self.interleave_size
        return owner, local

    @property
    def batch_size(self) -> int:
        return len(self._lengths)

    def append(self, request_values, index_values=None) -> None:
        """Append one KV and its matching Index-K entry per request."""
        import numpy as np

        values = np.asarray(request_values)
        if values.shape[0] != self.batch_size:
            raise ValueError("append batch does not match cache requests")
        if values.ndim < 2:
            raise ValueError("append values must include a feature axis")
        if not all(length == self._lengths[0] for length in self._lengths):
            raise ValueError("batched append requires equal request lengths")
        if values.shape[1] != 1:
            raise ValueError("append must contain exactly one token per request")
        indices = values if index_values is None else np.asarray(index_values)
        if indices.shape[:2] != values.shape[:2]:
            raise ValueError(
                "Index-K append must match request and token dimensions")
        if tuple(values.shape[2:]) != self._value_shape \
                or tuple(indices.shape[2:]) != self._index_shape:
            raise ValueError("append feature dimensions must match cache")
        position = self._lengths[0]
        for request in range(self.batch_size):
            owner, local = self.owner_local(request, position)
            key = (request, local)
            if key in self._shards[owner]:
                raise ValueError("append would overwrite an existing KV slot")
            self._shards[owner][key] = values[request, 0].copy()
            self._index_shards[owner][key] = indices[request, 0].copy()
            self._lengths[request] += 1

    def read(self, request: int):
        """Return the request's persistent history in global token order."""
        import numpy as np

        if request < 0 or request >= self.batch_size:
            raise IndexError("request id is outside the cache")
        entries = []
        for position in range(self._lengths[request]):
            owner, local = self.owner_local(request, position)
            key = (request, local)
            if key not in self._shards[owner]:
                raise RuntimeError(
                    f"missing KV entry request={request}, position={position}")
            entries.append(self._shards[owner][key])
        return np.stack(entries, axis=0)

    def read_index(self, request: int):
        """Return the request's Index-K history in global token order."""
        import numpy as np

        if request < 0 or request >= self.batch_size:
            raise IndexError("request id is outside the cache")
        entries = []
        for position in range(self._lengths[request]):
            owner, local = self.owner_local(request, position)
            key = (request, local)
            if key not in self._index_shards[owner]:
                raise RuntimeError(
                    f"missing Index-K entry request={request}, "
                    f"position={position}")
            entries.append(self._index_shards[owner][key])
        return np.stack(entries, axis=0)

    def read_shard(self, request: int, rank: int):
        """Read one owner's entries as sorted global positions and values."""
        import numpy as np

        if request < 0 or request >= self.batch_size:
            raise IndexError("request id is outside the cache")
        if not 0 <= rank < self.cp:
            raise ValueError("rank is outside the DCP group")
        positions = []
        values = []
        for position in range(self._lengths[request]):
            owner, local = self.owner_local(request, position)
            if owner != rank:
                continue
            key = (request, local)
            if key not in self._shards[rank]:
                raise RuntimeError(
                    f"missing KV entry request={request}, position={position}")
            positions.append(position)
            values.append(self._shards[rank][key])
        return positions, np.stack(values, axis=0)

    def read_index_shard(self, request: int, rank: int):
        """Read one owner's Index-K rows with their global cache positions."""
        import numpy as np

        if request < 0 or request >= self.batch_size:
            raise IndexError("request id is outside the cache")
        if not 0 <= rank < self.cp:
            raise ValueError("rank is outside the DCP group")
        positions = []
        values = []
        for position in range(self._lengths[request]):
            owner, local = self.owner_local(request, position)
            if owner == rank:
                key = (request, local)
                positions.append(position)
                values.append(self._index_shards[rank][key])
        return positions, np.stack(values, axis=0)

    def localize(self, request: int, positions, rank: int):
        """Return ``(global_position, owner_local_position)`` for local rows."""
        pairs = owner_localize_positions(
            positions, self.cp, rank, self.interleave_size)
        for position, _ in pairs:
            if position >= self._lengths[request]:
                raise IndexError("selected position is outside cache history")
        return pairs


def dcp_sparse_attention(main_cache: DCPKVCache, index_cache: DCPKVCache,
                         request: int, query, k: int,
                         index_query=None):
    """Execute global Index-K top-k, owner-local KV reads, and LSE merge.

    This numerical reference follows the DCP graph's dataflow: each rank
    scores its Index-K shard, the global top-k uses absolute cache positions,
    selected rows are mapped to their cyclic KV owners, and local attention
    partials are merged with stable LSE statistics.
    """
    if main_cache.cp != index_cache.cp:
        raise ValueError("main KV and Index-K caches need the same CP world")
    if request < 0 or request >= main_cache.batch_size \
            or request >= index_cache.batch_size:
        raise IndexError("request id is outside the cache")
    if main_cache._lengths[request] != index_cache._lengths[request]:
        raise ValueError("main KV and Index-K histories have different lengths")
    history_length = main_cache._lengths[request]
    if not 0 < k <= history_length:
        raise ValueError(
            f"top-k={k} outside [1, {history_length}] for decode attention")

    query = np.asarray(query, dtype=np.float64)
    index_query = np.asarray(
        query if index_query is None else index_query, dtype=np.float64)
    score_shards, index_shards = [], []
    for rank in range(index_cache.cp):
        positions, index_rows = index_cache.read_index_shard(request, rank)
        index_shards.append(np.asarray(positions, dtype=np.int64))
        score_shards.append(index_rows @ index_query)
    _, selected = exact_global_topk(score_shards, index_shards, k)

    selected_positions = [int(position) for position in selected]
    local_outputs, local_maxima, local_sums = [], [], []
    for rank in range(main_cache.cp):
        owned = main_cache.localize(request, selected_positions, rank)
        all_positions, all_values = main_cache.read_shard(request, rank)
        rows = dict(zip(all_positions, all_values))
        if not owned:
            local_outputs.append(
                np.zeros(main_cache._value_shape, dtype=np.float64))
            local_maxima.append([-np.inf])
            local_sums.append([0.0])
            continue
        local_values = np.stack([rows[position] for position, _ in owned])
        logits = local_values @ query
        local_max = np.max(logits)
        weights = np.exp(logits - local_max)
        local_outputs.append(
            (weights[:, None] * local_values).sum(axis=0) / weights.sum())
        local_maxima.append([local_max])
        local_sums.append([weights.sum()])
    output = stable_attention_merge(
        local_outputs, local_maxima, local_sums)
    return output, selected_positions


def verify_dcp_decode_reference(cp: int) -> dict:
    """Check two append-only reference steps; not full-model validation.

    Small synthetic, already-encoded caches exercise selection, ownership,
    persistence and LSE. The dense histories are maintained independently
    of DCPKVCache so an omitted or incorrect append cannot pass silently.
    This does not implement the model's compressed-cache update policy.
    """
    if cp <= 0:
        raise ValueError("CP must be positive")
    rng = np.random.default_rng(20261026)
    history = rng.normal(size=(2, max(7, cp), 5))
    index_history = rng.normal(size=(2, history.shape[1], 3))
    main_cache = DCPKVCache(cp, history)
    index_cache = DCPKVCache(cp, index_history)
    max_error = 0.0
    for _ in range(2):
        new_kv = rng.normal(size=(2, 1, 5))
        new_index = rng.normal(size=(2, 1, 3))
        main_cache.append(new_kv)
        index_cache.append(new_index)
        history = np.concatenate((history, new_kv), axis=1)
        index_history = np.concatenate((index_history, new_index), axis=1)
        for request in range(2):
            np.testing.assert_array_equal(main_cache.read(request), history[request])
            np.testing.assert_array_equal(
                index_cache.read_index(request), index_history[request])
            query = rng.normal(size=5)
            index_query = rng.normal(size=3)
            k = min(5, history.shape[1])
            positions = np.arange(history.shape[1])
            scores = index_history[request] @ index_query
            selected = np.lexsort((positions, -scores))[:k]
            rows = history[request, selected]
            logits = rows @ query
            weights = np.exp(logits - np.max(logits))
            expected = (weights[:, None] * rows).sum(axis=0) / weights.sum()
            actual, actual_ids = dcp_sparse_attention(
                main_cache, index_cache, request, query, k,
                index_query=index_query)
            np.testing.assert_array_equal(actual_ids, selected)
            np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    return {
        "scope": "synthetic-append-only-reference",
        "cp": cp, "steps": 2, "requests": 2,
        "max_abs_error": max_error,
        "full_model_multistep_validated": False,
    }


def _token_map(window: int, compressed: int, ratio: int, context: int):
    """Describe cache-position to source-token mapping as verified ranges."""
    result = []
    if window:
        result.append((
            0, window, context - min(window, context), context))
    if compressed:
        step = max(1, ratio)
        result.append((window, window + compressed, 0, step))
    return tuple(result)


def describe_dcp_kv_layouts(
    cp: int, batch_size: int, dp: int, context: int, dcp_enabled: bool,
    compress_ratios, window: int, stage_gpu_names,
    kv_dim: int, index_dim: int, kv_bytes: int, index_bytes: int,
):
    """Describe uniform CP cache ownership per layer and rank."""
    requests_per_group = batch_size // dp
    layouts = []
    for layer, ratio_value in enumerate(compress_ratios):
        ratio = int(ratio_value)
        compressed = context // ratio if ratio else 0
        total_cache = min(window, context) + compressed
        index_len = compressed if ratio == 4 else 0
        if (len(stage_gpu_names) == len(compress_ratios)
                and all(isinstance(value, (tuple, list))
                        for value in stage_gpu_names)):
            names = tuple(stage_gpu_names[layer])
        else:
            names = tuple(stage_gpu_names)
        for rank in range(cp):
            request_start, request_stop = 0, requests_per_group
            cache_start, cache_stop = 0, total_cache
            owners = (names[rank],)
            main_scheme = "cyclic" if dcp_enabled else "contiguous"
            requests = requests_per_group
            local_main_slots = (
                (total_cache + cp - 1 - rank) // cp
                if dcp_enabled else total_cache // cp)
            layouts.append(KVLayout(
                layer=layer,
                cache="main",
                dtype="fp8",
                request_start=request_start,
                request_stop=request_stop,
                cache_index_start=cache_start,
                cache_index_stop=cache_stop,
                owner_gpus=owners,
                replicas=1,
                bytes_per_replica=(
                    requests * local_main_slots * kv_dim
                    * kv_bytes),
                token_map=_token_map(
                    min(window, context), compressed, ratio, context),
                shard_scheme=main_scheme,
                rank=rank,
                shard_world=cp,
            ))
            if index_len:
                index_start, index_stop = 0, index_len
                index_replicated = dcp_enabled
                index_scheme = (
                    "replicated" if index_replicated else "contiguous")
                local_index_slots = (
                    index_len if index_replicated else index_len // cp)
                layouts.append(KVLayout(
                    layer=layer,
                    cache="index",
                    dtype="fp4",
                    request_start=request_start,
                    request_stop=request_stop,
                    cache_index_start=index_start,
                    cache_index_stop=index_stop,
                    owner_gpus=owners,
                    replicas=1,
                    bytes_per_replica=(
                        requests * local_index_slots * index_dim
                        * index_bytes),
                    token_map=((0, index_len, 0, 4),),
                    shard_scheme=index_scheme,
                    rank=rank,
                    shard_world=cp,
                    replicated=index_replicated,
                ))
    return layouts
