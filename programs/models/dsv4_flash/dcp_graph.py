"""Stateful DCP graph construction, sharing the legacy transformer body.

Only the opt-in route uses this module. Cache capacities are reserved once
for the complete decode wave. Attention and updates use logical positions,
not the padding. A second step consumes the first step's physical states.
"""

from dataclasses import replace
from math import lcm

from rooflang.language.graph import ComputeGraph
from rooflang.language.hardware.component import Memory
from rooflang.language.kernels.comm import AttentionLseReduce, Broadcast, CommKernel, Gather
from rooflang.language.kernels.forward import Nop, ReadInput, Sampling, Slice
from rooflang.language.kernels.identity import Spawn
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.programs.models.dsv4_flash import model, optimization
from rooflang.programs.models.dsv4_flash.config import (
    WINDOW, COMPRESS_RATIOS, KV_DIM, INDEX_HD, INDEX_H, INDEX_TOPK, D,
)
from rooflang.programs.models.dsv4_flash.dcp_ops import (
    DCPStateSpec, DCPStateLoad, DCPKVUpdate, DCPIndexUpdate, DCPGlobalTopK,
    DCPOwnerLocalRows, DCPAttentionPartial, DCPAttentionMerge,
)


class DCPCompressionState(Kernel):
    """Streaming compression cost and persistent projected accumulator.

    Projection FLOPs/weight shapes follow the existing prefill compressor.
    Encoded output rows are emitted only on a ratio boundary. RoofLang does
    not implement the learned compressor/quantizer numerically; numerical
    dataflow validation supplies encoded output rows at this boundary.
    """

    def __init__(self, batch, ratio, layer, emits):
        self.ratio, self.emits = ratio, emits
        self.width = (4 if ratio == 4 else 2) * KV_DIM
        super().__init__(inputs={
            "hidden": Tensor("bf16", (batch, 1, D)),
            "accumulator": Tensor("fp32", (batch, 1, self.width)),
        }, weights={"w": Tensor("fp32", (D, self.width), f"L{layer}_comp_w")},
            outputs={"state": Tensor("fp32", (batch, 1, self.width))})
        self.inplace_aliases = {"state": "accumulator"}
        if emits:
            self.outputs["kv_delta"] = Tensor("fp8", (batch, 1, KV_DIM))
            if ratio == 4:
                self.outputs["index_delta"] = Tensor("fp4", (batch, 1, INDEX_HD))
        self.dtype_ = "fp32"

    @property
    def flops(self):
        batch = self.inputs["hidden"].shape[0]
        return float(2 * batch * D * self.width +
                     (batch * KV_DIM * 6 if self.emits else 0))


def _copy_tensor(tensor):
    return Tensor(tensor.dtype, tensor.shape, tensor.weight_id)


def _wire(graph, placement, source, target, port):
    src, out = source
    graph.add_data_edge(src, target, {out: port})
    placement.set_tensor_memory(target.inputs[port],
                                placement.get_tensor_memory(src.outputs[out]))


def _add(graph, placement, kernel, device):
    graph.add_kernel(kernel)
    if not isinstance(kernel, CommKernel):
        placement.set_kernel_device(kernel, device)
    else:
        memory = placement._hardware.find_local_memory(device)
        for tensor in (*kernel.inputs.values(), *kernel.outputs.values()):
            placement.set_tensor_memory(tensor, memory)
    return kernel


def _fan(graph, placement, source, names, device):
    tensor = source[0].outputs[source[1]]
    fan = Spawn(len(names))
    fan.inputs = {"x": _copy_tensor(tensor)}
    fan.outputs = {name: _copy_tensor(tensor) for name in names}
    _add(graph, placement, fan, device)
    _wire(graph, placement, source, fan, "x")
    memory = placement.get_tensor_memory(fan.inputs["x"])
    for tensor in fan.outputs.values():
        placement.set_tensor_memory(tensor, memory)
    return {name: (fan, name) for name in names}


def _broadcast(graph, placement, source, devices):
    if len(devices) == 1:
        return [source]
    tensor = source[0].outputs[source[1]]
    comm = Broadcast(tensor.size_bytes, len(devices))
    comm.inputs = {"x": _copy_tensor(tensor)}
    comm.outputs = {f"o{r}": _copy_tensor(tensor) for r in range(len(devices))}
    graph.add_kernel(comm)
    _wire(graph, placement, source, comm, "x")
    for rank, device in enumerate(devices):
        placement.set_tensor_memory(comm.outputs[f"o{rank}"],
                                    placement._hardware.find_local_memory(device))
    return [(comm, f"o{rank}") for rank in range(len(devices))]


def append_attention_group(graph, placement, specs, devices, q_sources,
                           cache_sources, index_sources, new_kv_sources,
                           compressed_sources=None, new_index_sources=None):
    """Install the same explicit DCP DAG used by production and numerical checks."""
    partials, states, indices, records = [], [], [], []
    for rank, (spec, device) in enumerate(zip(specs, devices)):
        update = _add(graph, placement, DCPKVUpdate(spec), device)
        _wire(graph, placement, cache_sources[rank], update, "cache")
        _wire(graph, placement, new_kv_sources[rank], update, "new_kv")
        if spec.emits_compressed:
            _wire(graph, placement, compressed_sources[rank], update, "compressed_kv")
        state = _fan(graph, placement, (update, "updated"), ("read", "state"), device)
        states.append(state["state"])
        rows = _add(graph, placement, DCPOwnerLocalRows(spec), device)
        _wire(graph, placement, state["read"], rows, "cache")
        index_update, topk = None, None
        q_source = q_sources[rank]
        if spec.ratio == 4:
            index_update = _add(graph, placement, DCPIndexUpdate(spec), device)
            _wire(graph, placement, index_sources[rank], index_update, "cache")
            if spec.emits_compressed:
                _wire(graph, placement, new_index_sources[rank], index_update, "new_index")
            index_state = _fan(graph, placement, (index_update, "updated"),
                               ("read", "state"), device)
            indices.append(index_state["state"])
            q_fan = _fan(graph, placement, q_source, ("index", "attention"), device)
            topk = _add(graph, placement, DCPGlobalTopK(spec, min(INDEX_H, spec.heads)), device)
            _wire(graph, placement, q_fan["index"], topk, "q")
            _wire(graph, placement, index_state["read"], topk, "index_kv")
            _wire(graph, placement, (topk, "ids"), rows, "ids")
            q_source = q_fan["attention"]
        partial = _add(graph, placement, DCPAttentionPartial(spec, rows), device)
        _wire(graph, placement, q_source, partial, "q")
        _wire(graph, placement, (rows, "kv"), partial, "kv")
        _wire(graph, placement, (rows, "mask"), partial, "mask")
        partials.append(partial)
        records.append({"update": update, "index_update": index_update,
                        "topk": topk, "rows": rows, "partial": partial})
    merge = _add(graph, placement, DCPAttentionMerge(specs[0]), devices[0])
    for rank, partial in enumerate(partials):
        _wire(graph, placement, (partial, "y"), merge, f"i{rank}")
        _wire(graph, placement, (partial, "attn_stats"), merge, f"s{rank}")
    return (merge, "y"), states, indices, records


def _input_sources(graph, kernel):
    return {port: (edge.src, out) for edge in graph._in_edges(kernel)
            for out, port in edge.mapping.items()}


def _remove(graph, placement, kernel):
    graph.remove_kernel(kernel)
    placement._mapping.pop(kernel, None)


def _prune(graph, placement, protected):
    # Discard detached preload/barrier plumbing, preserving the transformer.
    allowed = (CommKernel, Spawn, Nop, ReadInput)
    queue = [k for k in graph.kernels if isinstance(k, allowed)
             and k not in protected and graph._dag.out_degree(k) == 0]
    while queue:
        kernel = queue.pop()
        if not graph._dag.has_node(kernel) or graph._dag.out_degree(kernel):
            continue
        predecessors = list(graph._dag.predecessors(kernel))
        _remove(graph, placement, kernel)
        queue.extend(k for k in predecessors if isinstance(k, allowed)
                     and k not in protected and graph._dag.out_degree(k) == 0)
    for kernel in graph.kernels:
        if isinstance(kernel, Spawn) and kernel not in protected:
            used = {out for edge in graph._out_edges(kernel) for out in edge.mapping}
            if used:
                kernel.outputs = {port: tensor for port, tensor in kernel.outputs.items()
                                  if port in used}
                kernel.world = len(kernel.outputs)


def _compression(graph, placement, layer, dp_rank, device, spec, previous, read_input):
    fields = layer._decode_copies["dp" if spec.cp > 1 else "cp_dp"]
    # Match the prefill compressor's input: pre-normalization layer hidden.
    fan = fields["bridge"][dp_rank]
    port = "dcp_compression"
    fan.outputs[port] = _copy_tensor(fan.inputs["x"])
    fan.world = len(fan.outputs)
    placement.set_tensor_memory(fan.outputs[port], placement.get_tensor_memory(fan.inputs["x"]))
    comp = _add(graph, placement, DCPCompressionState(
        spec.batch, spec.ratio, layer.sa.layer_id if hasattr(layer.sa, "layer_id") else 0,
        spec.emits_compressed), device)
    _wire(graph, placement, (fan, port), comp, "hidden")
    if previous is None:
        tensor = comp.inputs["accumulator"]
        root = ReadInput(tensor.n_elements, tensor.dtype)
        root.inputs = {"initial_state": _copy_tensor(tensor)}
        root.outputs = {"y": _copy_tensor(tensor)}
        _add(graph, placement, root, device)
        ssds = sorted((m for m in placement._hardware.nodes if isinstance(m, Memory)
                       and m.kind == "ssd" and m.name.split("-")[0] == device.name.split("-")[0]),
                      key=lambda m: m.name)
        if ssds:
            placement.set_tensor_memory(root.inputs["initial_state"], ssds[0])
        graph.add_control_edge(root, read_input)
        previous = (root, "y")
    _wire(graph, placement, previous, comp, "accumulator")
    return comp


def build_dcp_decode_graph(hw, *, batch_size, context, cp, dp, ep,
                           pp_partition, n_gpus, steps=2, graph_strategy=None,
                           capacity_steps=None):
    """Build an unrolled full transformer DAG with persistent DCP states."""
    if steps < 1 or context < 1 or cp < 1 or dp < 1 or ep < 1:
        raise ValueError("DCP decode requires positive context and steps")
    if cp * dp != ep:
        raise ValueError("this placement requires DP * CP == EP")
    # Reserve enough room before splitting, including integer CP shards.
    alignment = lcm(cp * 128, 128)
    reserve_steps = max(steps, capacity_steps or steps)
    padded_context = ((max(context + reserve_steps, WINDOW) + alignment - 1) // alignment) * alignment
    combined = ComputeGraph()
    placement = Placement(hardware=hw, graph=combined)
    previous_main, previous_index, previous_comp = {}, {}, {}
    previous_tokens = previous_end = first_read = None
    records, step_markers, all_states = [], [], []
    first_footprints = ()
    for step in range(steps):
        graph, layers, emb, token_read, kv_reads, output_head = model.declare_model(
            batch_size=batch_size, seq_prefill=padded_context, decode=True)
        graph, local_p = optimization.optimize_model_cluster_decode(
            graph, layers, hw, emb, token_read, kv_reads, output_head,
            seq_prefill=padded_context, cp=cp, dp=dp, ep=ep,
            pp_partition=pp_partition, n_gpus=n_gpus,
            graph_strategy=graph_strategy, dcp_enabled=True)
        first_read = first_read or token_read
        if not step:
            first_footprints = local_p.memory_footprints
        # Install each transformer wave before connecting persistent states;
        # previous-wave state producers already belong to the combined DAG.
        for kernel in graph.kernels:
            kernel.decode_step = step
            combined.add_kernel(kernel)
        for src, dst, attrs in graph._dag.edges(data=True):
            if attrs["mapping"]:
                combined.add_data_edge(src, dst, attrs["mapping"])
            else:
                combined.add_control_edge(src, dst)
        placement._mapping.update(local_p._mapping)
        placement._memory.update(local_p._memory)
        graph, local_p = combined, placement
        # Original barriers retain the prefix; replace them with updated state.
        for kernel in list(graph.kernels):
            if isinstance(kernel, Nop) and "decode_output" in kernel.inputs:
                _remove(graph, local_p, kernel)
        if step:
            for predecessor in list(graph._dag.predecessors(token_read)):
                if not graph._dag.edges[predecessor, token_read]["mapping"]:
                    graph.remove_control_edge(predecessor, token_read)
        main_states, index_states, comp_states = {}, {}, {}
        step_records = []
        for layer_id, layer in enumerate(layers):
            ratio = COMPRESS_RATIOS[layer_id]
            layer.sa.layer_id = layer_id
            old_copies = layer._decode_copies["cp_dp"]["sa"]
            replacements = []
            for dp_rank in range(dp):
                old = old_copies[dp_rank * cp:(dp_rank + 1) * cp]
                devices = [local_p.get_kernel_device(kernel).device for kernel in old]
                sources = [_input_sources(graph, kernel) for kernel in old]
                first = old[0]
                output_edges = graph._out_edges(first)
                old_merge = next((edge.dst for edge in output_edges
                                  if isinstance(edge.dst, AttentionLseReduce)), None)
                targets = [(edge.dst, dict(edge.mapping)) for edge in graph._out_edges(old_merge or first)
                           if "y" in edge.mapping]
                for kernel in old:
                    _remove(graph, local_p, kernel)
                if old_merge is not None:
                    _remove(graph, local_p, old_merge)
                fields = layer._decode_copies["dp" if cp > 1 else "cp_dp"]
                sink = fields["kv_sink"][dp_rank]
                _remove(graph, local_p, sink)
                norm = fields["kv_norm"][dp_rank]
                encode = Slice()
                encode.inputs = {"x": _copy_tensor(norm.outputs["y"])}
                encode.outputs = {"y": Tensor("fp8", (batch_size // dp, 1, KV_DIM))}
                _add(graph, local_p, encode, devices[0])
                _wire(graph, local_p, (norm, "y"), encode, "x")
                deltas = _broadcast(graph, local_p, (encode, "y"), devices)
                compressed_capacity = padded_context // ratio if ratio else 0
                spec = DCPStateSpec(context + step, WINDOW, ratio, compressed_capacity,
                                    cp, 0, batch_size // dp, KV_DIM, INDEX_HD,
                                    first.H, first.Hd, INDEX_TOPK)
                key = (layer_id, dp_rank)
                comp = None
                compressed_sources = new_index_sources = None
                if ratio:
                    comp = _compression(graph, local_p, layer, dp_rank, devices[0], spec,
                                        previous_comp.get(key), token_read)
                    comp_states[key] = (comp, "state")
                    if spec.emits_compressed:
                        compressed_sources = _broadcast(graph, local_p, (comp, "kv_delta"), devices)
                        if ratio == 4:
                            new_index_sources = _broadcast(graph, local_p, (comp, "index_delta"), devices)
                cache_sources = previous_main.get(key, [value["kv"] for value in sources])
                index_sources = previous_index.get(key, [value.get("index_kv") for value in sources])
                if not step:
                    # Include layout materialization in preload, not decode.
                    def materialize(sources):
                        result = []
                        for source, device in zip(sources, devices):
                            load = _add(graph, local_p, DCPStateLoad(
                                source[0].outputs[source[1]]), device)
                            _wire(graph, local_p, source, load, "x")
                            graph.add_control_edge(load, token_read)
                            result.append((load, "y"))
                        return result
                    cache_sources = materialize(cache_sources)
                    if ratio == 4:
                        index_sources = materialize(index_sources)
                output, states, indices, group_records = append_attention_group(
                    graph, local_p, [replace(spec, rank=rank) for rank in range(cp)], devices,
                    [value["q"] for value in sources], cache_sources, index_sources, deltas,
                    compressed_sources, new_index_sources)
                for record in group_records:
                    for kernel in record.values():
                        if kernel is not None:
                            kernel.layer_id, kernel.dp_rank = layer_id, dp_rank
                output[0].layer_id, output[0].dp_rank = layer_id, dp_rank
                for target, mapping in targets:
                    _wire(graph, local_p, output, target, mapping["y"])
                main_states[key] = states
                if indices:
                    index_states[key] = indices
                replacements.extend(record["partial"] for record in group_records)
                step_records.append({"layer": layer_id, "dp_rank": dp_rank,
                                     "output": output, "ranks": group_records,
                                     "compression": comp, "encoded_kv": encode})
            layer._decode_copies["cp_dp"]["sa"] = replacements
        def sampler_rank(kernel):
            name = local_p.get_kernel_device(kernel).device.name
            return int(name.split("-")[0][1:]), int(name.rsplit("-", 1)[1])

        samplers = sorted((kernel for kernel in graph.kernels if isinstance(kernel, Sampling)
                           and kernel.decode_step == step), key=sampler_rank)
        protected = {first_read, token_read, *samplers}
        protected.update(marker for pair in step_markers for marker in pair)
        # Keep exported state ports until the next wave consumes them.
        protected.update(source[0] for values in main_states.values() for source in values)
        protected.update(source[0] for values in index_states.values() for source in values)
        protected.update(source[0] for source in comp_states.values())
        # Protect state fan ports from trimming before the next step is linked.
        _prune(graph, local_p, protected)
        for values in [*main_states.values(), *index_states.values()]:
            for fan, port in values:
                if port not in fan.outputs:
                    fan.outputs[port] = _copy_tensor(fan.inputs["x"])
                    local_p.set_tensor_memory(fan.outputs[port], local_p.get_tensor_memory(fan.inputs["x"]))
        gather = Gather(batch_size * 4, dp, dim=0)
        gather.inputs = {f"i{rank}": _copy_tensor(kernel.outputs["y"])
                         for rank, kernel in enumerate(samplers)}
        gather.outputs = {"y": Tensor("int32", (batch_size, 1, 1))}
        final_device = local_p.get_kernel_device(samplers[0]).device
        _add(graph, local_p, gather, final_device)
        for rank, kernel in enumerate(samplers):
            _wire(graph, local_p, (kernel, "y"), gather, f"i{rank}")
        end = _add(graph, local_p, Nop(outputs={"done": Tensor("int32", (1,))}), final_device)
        graph.add_control_edge(gather, end)
        states_flat = [source for values in main_states.values() for source in values]
        states_flat += [source for values in index_states.values() for source in values]
        states_flat += list(comp_states.values())
        for source in states_flat:
            graph.add_control_edge(source[0], end)
        for kernel in graph.kernels:
            if not hasattr(kernel, "decode_step"):
                kernel.decode_step = step
        if step:
            _wire(combined, placement, previous_tokens, token_read, "tokens")
            combined.add_control_edge(previous_end, token_read)
        previous_main, previous_index, previous_comp = main_states, index_states, comp_states
        previous_tokens, previous_end = (gather, "y"), end
        records.append(step_records)
        step_markers.append((token_read, end))
        all_states = states_flat
    # A terminal barrier keeps all persistent allocations alive through output.
    terminal = Nop(inputs={f"state{i}": _copy_tensor(src.outputs[port])
                           for i, (src, port) in enumerate(all_states)},
                   outputs={"done": Tensor("int32", (1,))})
    final_device = placement.get_kernel_device(samplers[0]).device
    _add(combined, placement, terminal, final_device)
    for i, source in enumerate(all_states):
        _wire(combined, placement, source, terminal, f"state{i}")
    terminal.inputs["tokens"] = _copy_tensor(previous_tokens[0].outputs[previous_tokens[1]])
    _wire(combined, placement, previous_tokens, terminal, "tokens")
    combined.add_control_edge(previous_end, terminal)
    for footprint in first_footprints:
        placement.record_memory_footprint(footprint.memory, footprint.size_bytes, footprint.role)
    placement.metadata = {
        "dcp_enabled": True, "cp": cp, "dp": dp, "ep": ep,
        "pp_partition": list(pp_partition), "decode_steps": steps,
        "initial_context": context, "capacity_context": padded_context,
        "kv_layout": "ring-window-plus-cyclic-compressed",
        "index_kv_layout": "replicated-per-cp-rank",
        "compression_numeric_scope": "encoded-delta-boundary",
        "initial_cache_assumption": "prefix is pre-encoded in ring-window/cyclic-compressed layout",
        "prefill_to_decode_layout_conversion_included": False,
        "local_selection_timing": "uniform-expected-owner-count",
        "local_selection_allocation": "maximum-owner-skew",
    }
    combined.dcp_step_markers = step_markers
    combined.dcp_records = records
    combined.validate()
    placement.validate(combined)
    return combined, placement, first_read
