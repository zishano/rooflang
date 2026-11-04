# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Post-simulation timing analysis and report export.

The analysis consumes live ``SimulationResult`` objects so logical kernels and
multi-device collective slices remain distinguishable.  It never changes DES
scheduling or recomputes simulator costs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

from rooflang.language.kernels.comm import CommKernel
from rooflang.language.kernels.kernel import Kernel


@dataclass(frozen=True)
class RunConfig:
    model: str
    hardware: str
    stage: str
    batch_size: int
    context: int
    cp: int
    dp: int
    ep: int
    pp_partition: tuple[int, ...]
    measurement_start: str


@dataclass(frozen=True)
class AnalysisWindow:
    name: str
    start_us: float
    end_us: float
    origin_us: float

    @property
    def duration_us(self) -> float:
        return max(0.0, self.end_us - self.start_us)


@dataclass
class PerformanceReport:
    config: RunConfig
    window: AnalysisWindow
    summary: dict
    operators: list[dict]
    communications: list[dict]
    layers: list[dict]
    slices: list[dict]
    trace_events: list[dict]


_OPERATOR_FIELDS = (
    "logical_id", "semantic_key", "kernel_type", "operation_family",
    "plot_category", "bound", "layer_id", "operator_name", "provenance",
    "provenance_confidence", "source_layer_id", "destination_layer_id",
    "scope", "devices", "device_count", "stream", "start_us_absolute",
    "end_us_absolute", "start_us_relative", "end_us_relative", "duration_us",
    "clipped_duration_us", "window_fraction", "compute_time_us",
    "memory_time_us", "network_time_us", "local_elapsed_time_us",
    "network_elapsed_time_us", "attributed_local_elapsed_us",
    "attributed_network_elapsed_us", "exposed_local_us", "overlap_us",
    "exposed_network_us", "unattributed_us", "flops", "input_bytes",
    "weight_bytes", "loaded_weight_bytes", "output_bytes",
    "transferred_bytes",
)

_COMMUNICATION_FIELDS = (
    "logical_id", "semantic_key", "kernel_type", "communication_kind",
    "collective_type", "world", "participants", "layer_id",
    "source_layer_id", "destination_layer_id", "scope", "start_us_relative",
    "end_us_relative", "clipped_duration_us", "network_time_us",
    "network_elapsed_time_us", "exposed_network_us", "total_bytes",
    "transferred_bytes",
)

_LAYER_FIELDS = (
    "layer_id", "first_start_us", "last_end_us", "span_us",
    "logical_operator_count", "communication_count",
    "participating_device_count", "gpu_busy_us_total",
    "gpu_busy_us_mean_per_device", "gpu_busy_us_max_device",
    "local_path_work_us", "network_path_work_us", "overlap_work_us",
    "exposed_local_work_us", "exposed_network_work_us", "transferred_bytes",
    "flops",
)


def _json_value(value):
    if isinstance(value, Fraction):
        return int(value) if value.denominator == 1 else float(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_value(value), file, indent=2, ensure_ascii=False,
                  allow_nan=False)
        file.write("\n")


def _write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in rows:
            row = dict(record)
            for key, value in tuple(row.items()):
                if isinstance(value, (list, tuple, dict)):
                    row[key] = json.dumps(
                        _json_value(value), ensure_ascii=False, separators=(",", ":"))
            writer.writerow(row)


def _interval_union(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _walk_layer_value(value, path: str, graph_kernels: set[Kernel], seen: set[int]):
    if isinstance(value, Kernel):
        if value in graph_kernels:
            yield value, path
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_layer_value(
                item, f"{path}.{key}", graph_kernels, seen)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_layer_value(
                item, f"{path}[{index}]", graph_kernels, seen)
        return
    identity = id(value)
    if identity in seen or not hasattr(value, "__dict__"):
        return
    seen.add(identity)
    for key, item in vars(value).items():
        if isinstance(item, (Kernel, dict, list, tuple)):
            yield from _walk_layer_value(
                item, f"{path}.{key}", graph_kernels, seen)


def _direct_provenance(graph, layers: Sequence[object]) -> dict[Kernel, tuple[int, str]]:
    graph_kernels = set(graph.kernels)
    candidates: dict[Kernel, list[tuple[int, str]]] = defaultdict(list)
    for layer_id, layer in enumerate(layers):
        seen: set[int] = set()
        for kernel, path in _walk_layer_value(
                layer, f"layers[{layer_id}]", graph_kernels, seen):
            candidates[kernel].append((layer_id, path))
    return {
        kernel: min(paths, key=lambda item: (len(item[1]), item[1]))
        for kernel, paths in candidates.items()
    }


def _nearest_layers(graph, kernel: Kernel, direct: dict[Kernel, tuple[int, str]],
                    predecessors: bool) -> set[int]:
    queue = deque([(kernel, 0)])
    visited = {kernel}
    found_distance = None
    found: set[int] = set()
    while queue:
        current, distance = queue.popleft()
        if found_distance is not None and distance > found_distance:
            break
        if current is not kernel and current in direct:
            found_distance = distance
            found.add(direct[current][0])
            continue
        neighbors = graph._dag.predecessors(current) if predecessors \
            else graph._dag.successors(current)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    return found


def _operator_name(path: str | None, kernel_type: str) -> str:
    if not path:
        return kernel_type
    name = path.split(".")[-1]
    return name.split("[")[0]


def _classify(kernel: Kernel, operator_name: str, network_time_us: float) -> tuple[str, str]:
    kind = type(kernel).__name__
    lowered = f"{operator_name} {kind}".lower()
    if isinstance(kernel, CommKernel):
        return "communication", "communication"
    if kind in {"ReadInput", "Sampling", "Embedding"}:
        return "input_output", "input_output"
    if kind in {"Spawn", "Concat", "Nop"}:
        return "structural", "structural"
    if "dispatch" in lowered or "combine" in lowered or "routing" in lowered:
        return "moe_routing", "compute"
    if "expert" in lowered:
        return "moe_expert", "compute"
    if "attn" in lowered or "attention" in lowered or "kda" in lowered:
        return "attention", "compute"
    if "gemm" in lowered or "linear" in lowered or "projection" in lowered:
        return "gemm", "compute"
    if "norm" in lowered or "add" in lowered or "elementwise" in lowered:
        return "norm_residual", "compute"
    if network_time_us > 0:
        return "remote_compute", "compute"
    return "other_compute", "compute"


def _kernel_catalog(graph, layers: Sequence[object]) -> dict[Kernel, dict]:
    kernels = list(graph.topological_sort())
    logical_ids = {kernel: f"k{index:06d}" for index, kernel in enumerate(kernels)}
    direct = _direct_provenance(graph, layers)
    catalog = {}
    for kernel in kernels:
        kernel_type = type(kernel).__name__
        source_layers = _nearest_layers(graph, kernel, direct, True)
        destination_layers = _nearest_layers(graph, kernel, direct, False)
        source_layer = min(source_layers) if len(source_layers) == 1 else None
        destination_layer = min(destination_layers) \
            if len(destination_layers) == 1 else None
        if kernel in direct:
            layer_id, provenance = direct[kernel]
            confidence = "direct"
            scope = "layer"
        elif source_layer is not None and destination_layer is not None:
            layer_id = destination_layer
            scope = "inter_layer" if source_layer != destination_layer else "layer"
            provenance = f"inferred:{source_layer}->{destination_layer}"
            confidence = "inferred"
        elif destination_layer is not None:
            layer_id = destination_layer
            scope = "input_boundary"
            provenance = f"inferred:global->{destination_layer}"
            confidence = "inferred"
        elif source_layer is not None:
            layer_id = source_layer
            scope = "output_boundary"
            provenance = f"inferred:{source_layer}->global"
            confidence = "inferred"
        else:
            layer_id = None
            scope = "global"
            provenance = "global" if graph._dag.in_degree(kernel) == 0 \
                or graph._dag.out_degree(kernel) == 0 else "unresolved"
            confidence = provenance
        name = _operator_name(provenance if confidence == "direct" else None,
                              kernel_type)
        semantic = f"L{layer_id}.{name}.{kernel_type}" if layer_id is not None \
            else f"global.{name}.{kernel_type}"
        catalog[kernel] = {
            "logical_id": logical_ids[kernel],
            "semantic_key": semantic,
            "kernel_type": kernel_type,
            "layer_id": layer_id,
            "operator_name": name,
            "provenance": provenance,
            "provenance_confidence": confidence,
            "source_layer_id": source_layer,
            "destination_layer_id": destination_layer,
            "scope": scope,
        }
    return catalog


def _window(result, name: str) -> AnalysisWindow:
    if name == "measured":
        return AnalysisWindow(
            name="measured", start_us=result.measurement_start_us,
            end_us=result.total_time_us, origin_us=result.measurement_start_us)
    if name == "full":
        return AnalysisWindow(
            name="full", start_us=0.0, end_us=result.total_time_us,
            origin_us=0.0)
    raise ValueError(f"Unknown report window: {name}")


def _path_decomposition(duration: float, local: float, network: float) -> tuple[float, float, float, float]:
    if duration <= 0:
        return 0.0, 0.0, 0.0, 0.0
    local = min(max(0.0, local), duration)
    network = min(max(0.0, network), duration)
    overlap = max(0.0, local + network - duration)
    exposed_local = max(0.0, duration - network)
    exposed_network = max(0.0, duration - local)
    unattributed = max(0.0, duration - exposed_local - overlap - exposed_network)
    return exposed_local, overlap, exposed_network, unattributed


def build_performance_report(result, graph, placement, layers: Sequence[object], *,
                             config: RunConfig, window: str = "measured") -> PerformanceReport:
    analysis_window = _window(result, window)
    catalog = _kernel_catalog(graph, layers)
    entries_by_kernel: dict[Kernel, list] = defaultdict(list)
    for entry in result.trace:
        entries_by_kernel[entry.kernel].append(entry)

    operators: list[dict] = []
    slices: list[dict] = []
    trace_events: list[dict] = []
    communications: list[dict] = []
    excluded = partial = 0

    for kernel in graph.topological_sort():
        entries = entries_by_kernel.get(kernel, [])
        if not entries:
            continue
        first = entries[0]
        start = min(entry.start_us for entry in entries)
        end = max(entry.end_us for entry in entries)
        duration = max(0.0, end - start)
        clipped_start = max(start, analysis_window.start_us)
        clipped_end = min(end, analysis_window.end_us)
        clipped_duration = max(0.0, clipped_end - clipped_start)
        is_zero_inside = duration == 0 and analysis_window.start_us <= start <= analysis_window.end_us
        if clipped_duration <= 0 and not is_zero_inside:
            excluded += 1
            continue
        fraction = clipped_duration / duration if duration > 0 else 1.0
        if 0.0 < fraction < 1.0:
            partial += 1
        for entry in entries[1:]:
            for field in (
                "stream", "bound", "compute_time_us", "memory_time_us",
                "network_time_us", "local_elapsed_time_us",
                "network_elapsed_time_us",
            ):
                if getattr(entry, field) != getattr(first, field):
                    raise ValueError(
                        f"Inconsistent participant timing for {catalog[kernel]['logical_id']}: {field}")

        meta = catalog[kernel]
        family, plot_category = _classify(
            kernel, meta["operator_name"], first.network_time_us)
        exposed_local, overlap, exposed_network, unattributed = \
            _path_decomposition(
                duration, first.local_elapsed_time_us,
                first.network_elapsed_time_us)
        devices = tuple(sorted({entry.device.name for entry in entries}))
        record = {
            **meta,
            "operation_family": family,
            "plot_category": plot_category,
            "bound": first.bound.value,
            "devices": devices,
            "device_count": len(devices),
            "stream": first.stream,
            "start_us_absolute": start,
            "end_us_absolute": end,
            "start_us_relative": clipped_start - analysis_window.origin_us,
            "end_us_relative": clipped_end - analysis_window.origin_us,
            "duration_us": duration,
            "clipped_duration_us": clipped_duration,
            "window_fraction": fraction,
            "compute_time_us": first.compute_time_us,
            "memory_time_us": first.memory_time_us,
            "network_time_us": first.network_time_us,
            "local_elapsed_time_us": first.local_elapsed_time_us,
            "network_elapsed_time_us": first.network_elapsed_time_us,
            "attributed_local_elapsed_us": first.local_elapsed_time_us * fraction,
            "attributed_network_elapsed_us": first.network_elapsed_time_us * fraction,
            "exposed_local_us": exposed_local * fraction,
            "overlap_us": overlap * fraction,
            "exposed_network_us": exposed_network * fraction,
            "unattributed_us": unattributed * fraction,
            "flops": kernel.flops,
            "input_bytes": kernel.input_bytes,
            "weight_bytes": kernel.weight_bytes,
            "loaded_weight_bytes": kernel.loaded_weight_bytes,
            "output_bytes": kernel.output_bytes,
            "transferred_bytes": kernel.transferred_bytes,
        }
        operators.append(record)

        for entry in entries:
            slice_start = max(entry.start_us, analysis_window.start_us)
            slice_end = min(entry.end_us, analysis_window.end_us)
            if slice_end <= slice_start and not (
                    entry.end_us == entry.start_us
                    and analysis_window.start_us <= entry.start_us <= analysis_window.end_us):
                continue
            item = {
                "logical_id": meta["logical_id"],
                "semantic_key": meta["semantic_key"],
                "layer_id": meta["layer_id"],
                "operator_name": meta["operator_name"],
                "kernel_type": meta["kernel_type"],
                "operation_family": family,
                "plot_category": plot_category,
                "bound": first.bound.value,
                "device": entry.device.name,
                "stream": entry.stream,
                "start_us": slice_start - analysis_window.origin_us,
                "end_us": slice_end - analysis_window.origin_us,
                "duration_us": max(0.0, slice_end - slice_start),
                "network_time_us": entry.network_time_us,
            }
            slices.append(item)
            trace_events.append({
                "name": meta["semantic_key"],
                "cat": plot_category,
                "ph": "X",
                "ts": item["start_us"],
                "dur": item["duration_us"],
                "pid": entry.device.name,
                "tid": f"stream{entry.stream}",
                "args": {
                    "logical_id": meta["logical_id"],
                    "layer_id": meta["layer_id"],
                    "kernel_type": meta["kernel_type"],
                    "operator_name": meta["operator_name"],
                    "provenance": meta["provenance"],
                    "bound": first.bound.value,
                    "participants": devices,
                    "compute_time_us": first.compute_time_us,
                    "memory_time_us": first.memory_time_us,
                    "network_time_us": first.network_time_us,
                    "local_elapsed_time_us": first.local_elapsed_time_us,
                    "network_elapsed_time_us": first.network_elapsed_time_us,
                },
            })

        if isinstance(kernel, CommKernel):
            expected = tuple(sorted(
                device.name for device in placement.infer_comm_devices(kernel)))
            if expected != devices:
                raise ValueError(
                    f"Communication participant mismatch for {meta['logical_id']}: "
                    f"expected {expected}, traced {devices}")
            communication_kind = "collective" if len(devices) > 1 else "local_collective"
            total_bytes = getattr(kernel, "total_bytes", None)
            world = getattr(kernel, "world", len(devices))
        elif first.network_time_us > 0:
            communication_kind = "remote_memory"
            total_bytes = None
            world = len(devices)
        else:
            communication_kind = None
            total_bytes = None
            world = None
        if communication_kind is not None:
            communications.append({
                "logical_id": meta["logical_id"],
                "semantic_key": meta["semantic_key"],
                "kernel_type": meta["kernel_type"],
                "communication_kind": communication_kind,
                "collective_type": type(kernel).__name__ if isinstance(kernel, CommKernel) else "",
                "world": world,
                "participants": devices,
                "layer_id": meta["layer_id"],
                "source_layer_id": meta["source_layer_id"],
                "destination_layer_id": meta["destination_layer_id"],
                "scope": meta["scope"],
                "start_us_relative": record["start_us_relative"],
                "end_us_relative": record["end_us_relative"],
                "clipped_duration_us": clipped_duration,
                "network_time_us": first.network_time_us,
                "network_elapsed_time_us": first.network_elapsed_time_us,
                "exposed_network_us": record["exposed_network_us"],
                "total_bytes": total_bytes,
                "transferred_bytes": kernel.transferred_bytes if isinstance(kernel, CommKernel) else None,
            })

    operator_by_id = {record["logical_id"]: record for record in operators}
    device_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    stream_intervals: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    network_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for item in slices:
        interval = (item["start_us"], item["end_us"])
        device_intervals[item["device"]].append(interval)
        stream_intervals[(item["device"], item["stream"])].append(interval)
        if operator_by_id[item["logical_id"]]["network_time_us"] > 0:
            network_intervals[item["device"]].append(interval)

    window_duration = analysis_window.duration_us
    device_busy = {
        device: _interval_union(intervals)
        for device, intervals in sorted(device_intervals.items())
    }
    stream_busy = {
        f"{device}/stream{stream}": _interval_union(intervals)
        for (device, stream), intervals in sorted(stream_intervals.items())
    }
    network_busy = {
        device: _interval_union(intervals)
        for device, intervals in sorted(network_intervals.items())
    }
    busy_ratios = {
        device: (busy / window_duration if window_duration > 0 else 0.0)
        for device, busy in device_busy.items()
    }
    if any(value < -1e-12 or value > 1.0 + 1e-9 for value in busy_ratios.values()):
        raise ValueError(f"Invalid device busy ratio: {busy_ratios}")

    layer_rows = []
    layer_ids = sorted({
        record["layer_id"] for record in operators
        if record["layer_id"] is not None
    })
    slices_by_layer: dict[int, list[dict]] = defaultdict(list)
    operators_by_layer: dict[int, list[dict]] = defaultdict(list)
    for item in slices:
        if item["layer_id"] is not None:
            slices_by_layer[item["layer_id"]].append(item)
    for record in operators:
        if record["layer_id"] is not None:
            operators_by_layer[record["layer_id"]].append(record)
    for layer_id in layer_ids:
        layer_ops = operators_by_layer[layer_id]
        layer_slices = slices_by_layer[layer_id]
        intervals_by_device: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for item in layer_slices:
            intervals_by_device[item["device"]].append(
                (item["start_us"], item["end_us"]))
        busy_by_device = {
            device: _interval_union(intervals)
            for device, intervals in intervals_by_device.items()
        }
        first_start = min((item["start_us"] for item in layer_slices), default=0.0)
        last_end = max((item["end_us"] for item in layer_slices), default=0.0)
        layer_rows.append({
            "layer_id": layer_id,
            "first_start_us": first_start,
            "last_end_us": last_end,
            "span_us": max(0.0, last_end - first_start),
            "logical_operator_count": len(layer_ops),
            "communication_count": sum(
                record["network_time_us"] > 0 for record in layer_ops),
            "participating_device_count": len(busy_by_device),
            "gpu_busy_us_total": sum(busy_by_device.values()),
            "gpu_busy_us_mean_per_device": (
                sum(busy_by_device.values()) / len(busy_by_device)
                if busy_by_device else 0.0),
            "gpu_busy_us_max_device": max(busy_by_device.values(), default=0.0),
            "local_path_work_us": sum(
                record["attributed_local_elapsed_us"] for record in layer_ops),
            "network_path_work_us": sum(
                record["attributed_network_elapsed_us"] for record in layer_ops),
            "overlap_work_us": sum(record["overlap_us"] for record in layer_ops),
            "exposed_local_work_us": sum(
                record["exposed_local_us"] for record in layer_ops),
            "exposed_network_work_us": sum(
                record["exposed_network_us"] for record in layer_ops),
            "transferred_bytes": sum(record["transferred_bytes"] for record in layer_ops),
            "flops": sum(record["flops"] for record in layer_ops),
        })

    confidence = Counter(record["provenance_confidence"] for record in operators)
    families = Counter(record["operation_family"] for record in operators)
    bounds = Counter(record["bound"] for record in operators)
    summary = {
        "configuration": asdict(config),
        "window": asdict(analysis_window),
        "full_simulation_time_us": result.total_time_us,
        "kv_preload_time_us": result.measurement_start_us,
        "measured_decode_time_us": result.measured_time_us,
        "logical_operator_count": len(operators),
        "trace_slice_count": len(slices),
        "communication_count": len(communications),
        "collective_count": sum(
            item["communication_kind"] in {"collective", "local_collective"}
            for item in communications),
        "excluded_operator_count": excluded,
        "partially_clipped_operator_count": partial,
        "provenance_confidence": dict(confidence),
        "operation_families": dict(families),
        "bounds": dict(bounds),
        "device_busy_us": device_busy,
        "device_busy_ratio": busy_ratios,
        "stream_busy_us": stream_busy,
        "device_network_active_us": network_busy,
        "mean_device_busy_ratio": (
            sum(busy_ratios.values()) / len(busy_ratios)
            if busy_ratios else 0.0),
        "min_device_busy_ratio": min(busy_ratios.values(), default=0.0),
        "max_device_busy_ratio": max(busy_ratios.values(), default=0.0),
        "peak_memory_gb": {
            memory.name: size / 1e9 for memory, size in result.peak_memory.items()
        },
        "memory_footprints": [
            {
                "name": item.memory.name,
                "kind": item.memory.kind,
                "role": item.role,
                "bytes": item.size_bytes,
            }
            for item in result.memory_footprints
        ],
    }
    return PerformanceReport(
        config=config, window=analysis_window, summary=summary,
        operators=operators, communications=communications,
        layers=layer_rows, slices=slices, trace_events=trace_events)


def _git_source_info(root: Path) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_outputs(directory: Path, expected: Sequence[str], report: PerformanceReport) -> None:
    for name in expected:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty report output: {path}")
    for name in ("summary.json", "trace.json"):
        with (directory / name).open(encoding="utf-8") as file:
            json.load(file)
    with (directory / "operators.csv").open(encoding="utf-8") as file:
        operator_rows = sum(1 for _ in csv.DictReader(file))
    if operator_rows != len(report.operators):
        raise ValueError(
            f"operators.csv has {operator_rows} rows, expected {len(report.operators)}")
    communication_ids = [item["logical_id"] for item in report.communications]
    if len(communication_ids) != len(set(communication_ids)):
        raise ValueError("communications.csv contains duplicate logical IDs")
    valid_ids = {item["logical_id"] for item in report.operators}
    if any(event["args"]["logical_id"] not in valid_ids
           for event in report.trace_events):
        raise ValueError("trace.json references unknown logical IDs")


def publish_performance_report(
    report: PerformanceReport,
    final_dir: Path,
    *,
    repo_root: Path,
    focus_layer: int = 0,
    focus_device: str | None = None,
    keep_failed: bool = False,
) -> Path:
    """Write and atomically publish a complete timing report directory."""
    final_dir = final_dir.resolve()
    if final_dir.exists():
        raise FileExistsError(f"Report directory already exists: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = final_dir.parent / f".{final_dir.name}.tmp-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    started_at = datetime.now(timezone.utc).isoformat()
    expected = [
        "summary.json", "operators.csv", "communications.csv", "layers.csv",
        "trace.json", "timeline.png", "timeline.pdf", "layer_timeline.png",
        "layer_timeline.pdf", "layers.png", "layers.pdf",
    ]
    try:
        _write_json(staging / "summary.json", report.summary)
        _write_csv(staging / "operators.csv", report.operators, _OPERATOR_FIELDS)
        _write_csv(
            staging / "communications.csv", report.communications,
            _COMMUNICATION_FIELDS)
        _write_csv(staging / "layers.csv", report.layers, _LAYER_FIELDS)
        trace = {
            "traceEvents": report.trace_events,
            "otherData": report.summary,
        }
        _write_json(staging / "trace.json", trace)

        from rooflang.programs.experiments.performance_plots import (
            render_layer_analysis,
            render_layer_timeline,
            render_timeline,
        )
        render_timeline(report, staging / "timeline.png", staging / "timeline.pdf")
        render_layer_timeline(
            report, staging / "layer_timeline.png",
            staging / "layer_timeline.pdf", layer_id=focus_layer,
            device=focus_device)
        render_layer_analysis(
            report, staging / "layers.png", staging / "layers.pdf")
        _validate_outputs(staging, expected, report)

        files = []
        for name in expected:
            path = staging / name
            files.append({
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "run_id": final_dir.name,
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": sys.argv,
            "config": asdict(report.config),
            "window": report.window.name,
            "source": _git_source_info(repo_root),
            "files": files,
        }
        _write_json(staging / "manifest.json", manifest)
        if not (staging / "manifest.json").stat().st_size:
            raise ValueError("Empty manifest")
        os.replace(staging, final_dir)
        return final_dir
    except Exception as error:
        if keep_failed:
            failed_root = final_dir.parent / "_failed"
            failed_root.mkdir(exist_ok=True)
            failed_dir = failed_root / f"{final_dir.name}-{uuid.uuid4().hex[:8]}"
            _write_json(staging / "manifest.json", {
                "schema_version": 1,
                "status": "failed",
                "run_id": final_dir.name,
                "error": f"{type(error).__name__}: {error}",
                "config": asdict(report.config),
            })
            os.replace(staging, failed_dir)
        else:
            shutil.rmtree(staging, ignore_errors=True)
        raise
