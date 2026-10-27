"""Numerical checks of actual DCP graph nodes at encoded projection boundaries.

The DES itself is a timing/memory simulator, not a model tensor executor.
Here the exact state/selection/attention DAG is evaluated with independent
dense histories. Learned projection, compression and quantization outputs
are supplied as encoded tensors, so this is not pretrained-model validation.
"""

import numpy as np

from rooflang.language.graph import ComputeGraph
from rooflang.language.kernels.forward import Nop
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.kernels.identity import Spawn
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.programs.models.dsv4_flash.dcp_graph import (
    _input_sources, _add, _wire, append_attention_group,
)
from rooflang.programs.models.dsv4_flash.dcp_ops import DCPStateSpec
from rooflang.programs.presets.h200 import H200Cluster


def validate_dcp_graph(graph, *, seed=20261026, state=None):
    """Evaluate every DCP group and step, including persistent state edges."""
    reused = state is not None
    state = state if reused else {}
    rng = state.setdefault("rng", np.random.default_rng(seed))
    dense = state.setdefault("dense", {})
    shards = state.setdefault("shards", {})
    seeds, values = {}, {}
    max_error, checked = 0.0, 0

    def source(kernel, port):
        return _input_sources(graph, kernel)[port]

    def inject_input(kernel, port, value):
        key = source(kernel, port)
        if key in seeds:
            np.testing.assert_array_equal(seeds[key], value)
        seeds[key] = value

    def evaluate(kernel, port):
        key = (kernel, port)
        if key in seeds:
            return seeds[key]
        if key not in values:
            inputs = {name: evaluate(*src) for name, src in _input_sources(graph, kernel).items()}
            if isinstance(kernel, Spawn):
                result = {name: inputs["x"] for name in kernel.outputs}
            elif hasattr(kernel, "evaluate"):
                result = kernel.evaluate(inputs)
            else:
                raise ValueError(f"unprovided numerical boundary: {type(kernel).__name__}.{port}")
            values.update({(kernel, name): value for name, value in result.items()})
        return values[key]

    for step_records in graph.dcp_records:
        for record in step_records:
            ranks = record["ranks"]
            spec = ranks[0]["update"].spec
            key = record["layer"], record["dp_rank"]
            if key not in dense:
                main = rng.normal(size=(spec.batch, spec.capacity, spec.kv_dim))
                index = rng.normal(size=(spec.batch, spec.compressed_capacity, spec.index_dim))
                dense[key] = main, index
                shards[key] = [(main[:, rank::spec.cp].copy(), index.copy())
                               for rank in range(spec.cp)]
                initialize = True
            else:
                initialize = reused
            if initialize:
                for rank, nodes in enumerate(ranks):
                    inject_input(nodes["update"], "cache", shards[key][rank][0])
                    if spec.ratio == 4:
                        inject_input(nodes["index_update"], "cache", shards[key][rank][1])
            main, index = dense[key]
            new_kv = rng.normal(size=(spec.batch, 1, spec.kv_dim))
            q = rng.normal(size=(spec.batch, 1, spec.heads * spec.head_dim))
            main[:, spec.context % spec.window] = new_kv[:, 0]
            compressed = new_index = None
            if spec.emits_compressed:
                compressed = rng.normal(size=(spec.batch, 1, spec.kv_dim))
                main[:, spec.window + spec.compressed - 1] = compressed[:, 0]
                if spec.ratio == 4:
                    new_index = rng.normal(size=(spec.batch, 1, spec.index_dim))
                    index[:, spec.compressed - 1] = new_index[:, 0]
            for nodes in ranks:
                inject_input(nodes["update"], "new_kv", new_kv)
                inject_input(nodes["partial"], "q", q)
                if spec.emits_compressed:
                    inject_input(nodes["update"], "compressed_kv", compressed)
                if spec.ratio == 4:
                    inject_input(nodes["topk"], "q", q)
                    if spec.emits_compressed:
                        inject_input(nodes["index_update"], "new_index", new_index)
            # Compute the reference independently in global physical-slot order.
            expected = np.zeros((spec.batch, spec.heads, spec.head_dim))
            queries = q.reshape(spec.batch, spec.heads, spec.head_dim)
            for batch in range(spec.batch):
                if spec.ratio == 4:
                    heads = ranks[0]["topk"].index_heads
                    iq = queries[batch, :heads, :spec.index_dim]
                    scores = np.maximum(iq @ index[batch, :spec.compressed].T, 0).sum(axis=0)
                    selected = np.lexsort((np.arange(spec.compressed), -scores))[:spec.selected_compressed]
                    for nodes in ranks:
                        np.testing.assert_array_equal(evaluate(nodes["topk"], "ids")[batch], selected)
                else:
                    selected = np.arange(spec.compressed)
                # Do not call DCPStateSpec.window_slots in the reference.
                positions = [p % spec.window for p in range(
                    max(0, spec.context + 1 - spec.window), spec.context + 1)]
                positions += [spec.window + int(i) for i in selected]
                rows = main[batch, positions]
                logits = queries[batch] @ rows.T
                weights = np.exp(logits - logits.max(axis=1, keepdims=True))
                expected[batch] = (weights @ rows) / weights.sum(axis=1, keepdims=True)
            actual = evaluate(*record["output"]).reshape(expected.shape)
            np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
            for rank, nodes in enumerate(ranks):
                np.testing.assert_array_equal(evaluate(nodes["update"], "updated"), main[:, rank::spec.cp])
                if spec.ratio == 4:
                    np.testing.assert_array_equal(evaluate(nodes["index_update"], "updated"), index)
                if reused:
                    shards[key][rank] = (
                        evaluate(nodes["update"], "updated").copy(),
                        evaluate(nodes["index_update"], "updated").copy()
                        if spec.ratio == 4 else index.copy())
            checked += 1
    state["steps"] = state.get("steps", 0) + len(graph.dcp_records)
    return {
        "scope": "actual-dcp-graph-at-encoded-projection-boundaries",
        "steps": state["steps"], "groups_checked": checked,
        "max_abs_error": max_error,
        "state_topk_owner_attention_lse_validated": True,
        "pretrained_model_numerics_validated": False,
    }


def validate_dcp_startup(cp):
    """Exercise production DCP DAG operators with small encoded tensors.

    Cover sliding-window overwrite, a compressed append, and an empty top-k
    owner at the requested CP degree. This bounded check avoids allocating
    a full numerical copy of a 64K/1M model simulation's resident KV cache.
    """
    if cp < 1:
        raise ValueError("CP must be positive")
    from dataclasses import replace

    hw = H200Cluster(n_nodes=(cp + 7) // 8)
    devices = sorted((node for node in hw.nodes if getattr(node, "kind", None) == "gpu"),
                     key=lambda node: (int(node.name.split("-")[0][1:]),
                                       int(node.name.rsplit("-", 1)[1])))[:cp]
    graph = ComputeGraph()
    placement = Placement(hardware=hw, graph=graph)
    graph.dcp_records = [[], []]

    def root(shape, dtype, device):
        kernel = _add(graph, placement, Kernel(outputs={"y": Tensor(dtype, shape)}), device)
        return kernel, "y"

    for layer, ratio in enumerate((0, 4, 128)):
        context = 127
        window = ((max(cp, 8) + cp - 1) // cp) * cp
        capacity = (((context + 2) // ratio + cp - 1) // cp) * cp if ratio else 0
        spec = DCPStateSpec(context, window, ratio, capacity, cp, 0, 1, 8, 4, 2, 8, 1)
        states = [root((1, spec.local_capacity, 8), "fp8", device) for device in devices]
        indices = ([root((1, capacity, 4), "fp4", device) for device in devices]
                   if ratio == 4 else [None] * cp)
        for step in range(2):
            current = replace(spec, context=context + step)
            queries = [root((1, 1, 16), "bf16", device) for device in devices]
            deltas = [root((1, 1, 8), "fp8", device) for device in devices]
            compressed = ([root((1, 1, 8), "fp8", device) for device in devices]
                          if current.emits_compressed else None)
            index_deltas = ([root((1, 1, 4), "fp4", device) for device in devices]
                            if current.emits_compressed and ratio == 4 else None)
            output, states, next_indices, records = append_attention_group(
                graph, placement, [replace(current, rank=rank) for rank in range(cp)],
                devices, queries, states, indices, deltas, compressed, index_deltas)
            if next_indices:
                indices = next_indices
            graph.dcp_records[step].append({
                "layer": layer, "dp_rank": 0, "ranks": records, "output": output,
            })
        final_states = states + (indices if ratio == 4 else [])
        terminal = _add(graph, placement, Nop(inputs={
            f"state{i}": Tensor(source[0].outputs[source[1]].dtype,
                                 source[0].outputs[source[1]].shape)
            for i, source in enumerate(final_states)
        }), devices[0])
        for i, source in enumerate(final_states):
            _wire(graph, placement, source, terminal, f"state{i}")
    graph.validate()
    placement.validate(graph)
    report = validate_dcp_graph(graph)
    report["scope"] = "production-dcp-operators-at-small-encoded-boundaries"
    report["cp"] = cp
    return report
