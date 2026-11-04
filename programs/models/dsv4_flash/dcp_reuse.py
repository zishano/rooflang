"""Reuse one placed transformer graph across a sequential decode wave.

The DES models the recurrent cache as persistent external state between
invocations. The same tensor descriptors and memory locations are retained;
only initial preloads are skipped after the first invocation. Dynamic DCP
parameters and occasional compression edges are refreshed locally.
"""

import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import networkx as nx

from rooflang.language.kernels.comm import CommKernel
from rooflang.language.tensor import Tensor
from rooflang.runtime.simulator import Simulator, SimulationResult
from rooflang.runtime.trace_export import export_trace
from rooflang.programs.models.dsv4_flash.dcp_graph import (
    build_dcp_decode_graph, _input_sources, _broadcast, _wire, _remove,
)
from rooflang.programs.models.dsv4_flash.dcp_ops import (
    DCPKVUpdate, DCPIndexUpdate, DCPGlobalTopK, DCPOwnerLocalRows,
    DCPAttentionPartial,
)


def _update_descriptor(kernel, fresh, placement):
    """Refresh dimensions/costs without replacing nodes or persistent tensors."""
    for name, value in vars(fresh).items():
        if name not in ("inputs", "outputs", "weights"):
            setattr(kernel, name, value)
    for category in ("inputs", "outputs"):
        existing = getattr(kernel, category)
        desired = getattr(fresh, category)
        for port in tuple(existing):
            if port not in desired:
                placement._memory.pop(existing.pop(port), None)
        for port, tensor in desired.items():
            if port not in existing:
                existing[port] = tensor
            else:
                existing[port].shape = tensor.shape
                existing[port].dtype = tensor.dtype


class ReusableDCP:
    """Build/split/place once; retain the same compute and state tensor IDs."""

    def __init__(self, hw, **kwargs):
        started = perf_counter()
        self.steps = kwargs.pop("steps", 1)
        if self.steps < 1:
            raise ValueError("decode steps must be positive")
        self.context = kwargs["context"]
        self.graph, self.placement, self.reader = build_dcp_decode_graph(
            hw, **kwargs, steps=1, capacity_steps=self.steps)
        self.hardware = hw
        self.records = self.graph.dcp_records[0]
        self.start, self.end = self.graph.dcp_step_markers[0]
        self.preloads = frozenset(nx.ancestors(self.graph._dag, self.reader))
        self.simulator = Simulator(self.graph, self.placement, hw, measurement_start=self.reader)
        self.position = 0
        self.placement.metadata.update(decode_steps=self.steps, execution="reused-graph")
        self.build_seconds = perf_counter() - started

    def _toggle_compression(self, record, emits):
        comp = record["compression"]
        if comp is None or comp.emits == emits:
            return set()
        graph, placement = self.graph, self.placement
        changed = {comp}
        devices = [placement.get_kernel_device(rank["partial"]).device
                   for rank in record["ranks"]]
        # Remove only the optional delta broadcasts and their data edges.
        for port in ("kv_delta", "index_delta"):
            if port not in comp.outputs:
                continue
            for edge in list(graph._out_edges(comp)):
                if port not in edge.mapping:
                    continue
                if isinstance(edge.dst, CommKernel):
                    changed.add(edge.dst)
                    for tensor in (*edge.dst.inputs.values(), *edge.dst.outputs.values()):
                        placement._memory.pop(tensor, None)
                    _remove(graph, placement, edge.dst)
                else:  # CP=1 connects the compressor directly to the update.
                    graph._remove_data_mapping(comp, edge.dst, port)
            placement._memory.pop(comp.outputs.pop(port), None)
        comp.emits = emits
        if emits:
            spec = record["ranks"][0]["update"].spec
            comp.outputs["kv_delta"] = Tensor("fp8", (spec.batch, 1, spec.kv_dim))
            if spec.ratio == 4:
                comp.outputs["index_delta"] = Tensor("fp4", (spec.batch, 1, spec.index_dim))
            memory = self.hardware.find_local_memory(devices[0])
            for port in ("kv_delta", "index_delta"):
                if port not in comp.outputs:
                    continue
                placement.set_tensor_memory(comp.outputs[port], memory)
                sources = _broadcast(graph, placement, (comp, port), devices)
                for source, rank in zip(sources, record["ranks"]):
                    target = rank["update"] if port == "kv_delta" else rank["index_update"]
                    target_port = "compressed_kv" if port == "kv_delta" else "new_index"
                    _wire(graph, placement, source, target, target_port)
                    changed.add(source[0])
        return changed

    def configure(self, step):
        if not 0 <= step < self.steps:
            raise ValueError("decode step outside reserved wave")
        changed = {self.reader}
        for record in self.records:
            current = record["ranks"][0]["update"].spec
            emits = bool(current.ratio and (self.context + step + 1) % current.ratio == 0)
            # Drop old optional edges before updating the affected port sets.
            if record["compression"] is not None and record["compression"].emits and not emits:
                changed.update(self._toggle_compression(record, False))
            for rank in record["ranks"]:
                spec = replace(rank["update"].spec, context=self.context + step)
                _update_descriptor(rank["update"], DCPKVUpdate(spec), self.placement)
                if rank["index_update"] is not None:
                    _update_descriptor(rank["index_update"], DCPIndexUpdate(spec), self.placement)
                    _update_descriptor(rank["topk"], DCPGlobalTopK(spec, rank["topk"].index_heads), self.placement)
                _update_descriptor(rank["rows"], DCPOwnerLocalRows(spec), self.placement)
                _update_descriptor(rank["partial"], DCPAttentionPartial(spec, rank["rows"]), self.placement)
                changed.update(node for node in rank.values() if node is not None)
            changed.update(self._toggle_compression(record, emits))
            record["output"][0].spec = replace(record["output"][0].spec,
                                                context=self.context + step)
        if step:
            # Previous sampled tokens are already on the final-stage device.
            terminal = next(kernel for kernel in self.graph._dag
                            if "tokens" in kernel.inputs and "state0" in kernel.inputs)
            previous_tokens = _input_sources(self.graph, terminal)["tokens"]
            tensor = previous_tokens[0].outputs[previous_tokens[1]]
            self.placement.set_tensor_memory(self.reader.inputs["tokens"],
                                             self.placement.get_tensor_memory(tensor))
        self.simulator.invalidate_kernels(changed)
        self.position = step
        return changed

    def run_step(self, step):
        self.configure(step)
        # No global graph transforms or placements happen here.
        return self.simulator.run(precompleted=self.preloads if step else (), reuse_plan=True)


def simulate_reused_decode(plan, trace_path=None, *, trace_mode="none",
                           trace_steps=None, progress=None):
    """Run every step; emit a compact summary or separate per-step traces.

    A trace is exported before any graph mutation, preserving the original
    dimensions/costs of that step without retaining N copies of its kernels.
    """
    if trace_mode not in ("none", "first", "all"):
        raise ValueError("DCP trace mode must be none, first or all")
    selected_steps = None if trace_steps is None else set(trace_steps)
    if selected_steps is not None and any(step < 1 or step > plan.steps
                                          for step in selected_steps):
        raise ValueError("DCP trace steps must be within the decode wave")
    if selected_steps is not None and trace_mode != "none":
        raise ValueError("choose either --dcp-trace or --dcp-trace-steps")
    wall_started = perf_counter()
    step_times, wall_steps, trace_files = [], [], []
    peak_memory = {}
    preload = total = export_seconds = 0.0
    event_count = 0
    max_kernels = 0
    for step in range(plan.steps):
        started = perf_counter()
        result = plan.run_step(step)
        wall_steps.append(perf_counter() - started)
        if not step:
            preload = result.measurement_start_us
        duration = result.measured_time_us
        step_times.append(duration)
        for memory, size in result.peak_memory.items():
            peak_memory[memory] = max(peak_memory.get(memory, 0.0), size)
        event_count += len(result.trace)
        max_kernels = max(max_kernels, len(plan.graph._dag))
        export_step = (
            trace_mode == "all" or trace_mode == "first" and step == 0
            or selected_steps is not None and step + 1 in selected_steps)
        if trace_path and export_step:
            export_started = perf_counter()
            base = Path(trace_path)
            target = base.with_name(f"{base.stem}_step{step + 1}{base.suffix}")
            # The kernel objects are reused; serialize this step immediately.
            for kernel in plan.graph._dag:
                kernel.decode_step = step
            result.dcp_metadata = {**plan.placement.metadata, "step": step,
                                   "context_before_step": plan.context + step,
                                   "timeline_offset_us": total if not step else preload + total}
            export_trace(result, str(target))
            trace_files.append(str(target))
            export_seconds += perf_counter() - export_started
        total += duration
        if progress:
            progress(step + 1, duration, wall_steps[-1])
    metadata = {
        **plan.placement.metadata, "step_times_us": step_times,
        "graph_builds": 1, "graph_kernels": len(plan.graph._dag),
        "max_graph_kernels": max_kernels, "events_simulated": event_count,
        "trace_mode": "selected" if selected_steps is not None else trace_mode,
        "trace_steps": sorted(selected_steps) if selected_steps is not None else None,
        "trace_files": trace_files,
        "wall_build_seconds": plan.build_seconds, "wall_step_seconds": wall_steps,
        "wall_simulation_seconds": sum(wall_steps), "wall_export_seconds": export_seconds,
        "wall_run_seconds": perf_counter() - wall_started,
    }
    combined = SimulationResult([], preload + total, peak_memory, preload,
                                plan.placement.memory_footprints)
    combined.dcp_metadata = metadata
    if trace_path:
        # Keep the conventional JSON entrypoint, with a bounded-size manifest.
        manifest = {"traceEvents": [], "otherData": {
            "total_time_us": combined.total_time_us,
            "measurement_start_us": preload, "measured_time_us": total,
            "dcp": metadata,
            "peak_memory": [{"name": memory.name, "bytes": size}
                            for memory, size in peak_memory.items()],
        }}
        Path(trace_path).write_text(json.dumps(manifest, indent=2) + "\n")
    return combined
