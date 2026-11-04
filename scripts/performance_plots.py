# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Static plots for post-simulation performance reports."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


COLORS = {
    "compute": "#2a78d6",
    "communication": "#eb6834",
    "input_output": "#1baf7a",
    "structural": "#898781",
}
GRID = "#e1e0d9"
TEXT = "#0b0b0b"
MUTED = "#52514e"


def _save(figure, png_path: Path, pdf_path: Path) -> None:
    figure.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _legend() -> list[Line2D]:
    return [
        Line2D([0], [0], color=COLORS["compute"], lw=7, label="local compute / memory"),
        Line2D([0], [0], color=COLORS["communication"], lw=7,
               label="explicit communication"),
        Line2D([0], [0], color=COLORS["input_output"], lw=7,
               label="input / output"),
        Line2D([0], [0], color=MUTED, lw=2, linestyle="--",
               label="contains network work"),
    ]


def _style_axis(axis) -> None:
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=MUTED)
    axis.xaxis.label.set_color(TEXT)
    axis.yaxis.label.set_color(TEXT)
    axis.title.set_color(TEXT)


def _line_collections(axis, slices, lane_index, linewidth=12):
    grouped = defaultdict(list)
    network_segments = []
    for item in slices:
        if item["duration_us"] <= 0:
            continue
        lane = lane_index[(item["device"], item["stream"])]
        segment = [(item["start_us"], lane), (item["end_us"], lane)]
        grouped[item["plot_category"]].append(segment)
        if item["network_time_us"] > 0 and item["plot_category"] != "communication":
            network_segments.append(segment)
    for category, segments in grouped.items():
        collection = LineCollection(
            segments, colors=COLORS.get(category, COLORS["compute"]),
            linewidths=linewidth, capstyle="butt", rasterized=True)
        axis.add_collection(collection)
    if network_segments:
        axis.add_collection(LineCollection(
            network_segments, colors=MUTED, linewidths=max(1.0, linewidth * 0.22),
            linestyles="dashed", rasterized=True))


def render_timeline(report, png_path: Path, pdf_path: Path) -> None:
    """Render every observed device/stream lane over the selected window."""
    lanes = sorted({(item["device"], item["stream"]) for item in report.slices})
    lane_index = {lane: index for index, lane in enumerate(lanes)}
    height = max(4.0, 0.45 * len(lanes) + 2.0)
    figure, axis = plt.subplots(figsize=(16, height))
    _line_collections(axis, report.slices, lane_index, linewidth=13)
    axis.set_yticks(range(len(lanes)))
    axis.set_yticklabels([f"{device} / stream{stream}" for device, stream in lanes],
                         fontsize=8)
    axis.set_ylim(-0.8, max(0.8, len(lanes) - 0.2))
    axis.set_xlim(0, max(report.window.duration_us, 1.0))
    axis.set_xlabel(f"time relative to {report.window.name} window (μs)")
    axis.set_ylabel("actual simulated lanes")
    axis.set_title(
        f"{report.config.model} {report.config.stage} · {report.config.hardware} · "
        f"batch {report.config.batch_size} · CP/DP/EP "
        f"{report.config.cp}/{report.config.dp}/{report.config.ep}")
    axis.legend(handles=_legend(), loc="upper right", frameon=True, ncol=2)
    _style_axis(axis)
    figure.tight_layout()
    _save(figure, png_path, pdf_path)


def _resolve_device(devices: list[str], requested: str | None) -> str | None:
    if not devices:
        return None
    if requested is None:
        return devices[0]
    if requested in devices:
        return requested
    if requested.isdigit():
        matches = [name for name in devices if name.rsplit("-", 1)[-1] == requested]
        if len(matches) == 1:
            return matches[0]
    raise ValueError(
        f"Unknown focus device {requested!r}; choose one of: {', '.join(devices)}")


def render_layer_timeline(report, png_path: Path, pdf_path: Path, *,
                          layer_id: int = 0, device: str | None = None) -> None:
    """Render an Nsight-like focused layer/rank timeline and work breakdown."""
    devices = sorted({item["device"] for item in report.slices})
    selected_device = _resolve_device(devices, device)
    selected = [
        item for item in report.slices
        if item["layer_id"] == layer_id and item["device"] == selected_device
    ]
    if not selected:
        raise ValueError(
            f"No timeline slices for layer {layer_id} on {selected_device}")
    streams = sorted({item["stream"] for item in selected})
    lane_index = {(selected_device, stream): index for index, stream in enumerate(streams)}
    start = min(item["start_us"] for item in selected)
    end = max(item["end_us"] for item in selected)

    operators = {
        record["logical_id"]: record for record in report.operators
        if record["layer_id"] == layer_id
        and selected_device in record["devices"]
    }
    family_work = defaultdict(lambda: [0.0, 0.0, 0.0])
    for record in operators.values():
        bucket = family_work[record["operation_family"]]
        bucket[0] += record["exposed_local_us"]
        bucket[1] += record["overlap_us"]
        bucket[2] += record["exposed_network_us"]

    figure, (timeline_axis, work_axis) = plt.subplots(
        2, 1, figsize=(15, 6.8), gridspec_kw={"height_ratios": [2.1, 1.2]})
    _line_collections(timeline_axis, selected, lane_index, linewidth=18)
    timeline_axis.set_yticks(range(len(streams)))
    timeline_axis.set_yticklabels([f"stream{stream}" for stream in streams])
    timeline_axis.set_ylim(-0.7, max(0.7, len(streams) - 0.3))
    pad = max((end - start) * 0.03, 0.01)
    timeline_axis.set_xlim(start - pad, end + pad)
    timeline_axis.axvline(end, color=MUTED, linestyle="--", linewidth=1.2,
                          label="layer completion")
    timeline_axis.set_xlabel(f"time relative to {report.window.name} start (μs)")
    timeline_axis.set_ylabel("actual stream")
    timeline_axis.set_title(
        f"Layer {layer_id} · {selected_device} · simulated kernel timeline")
    timeline_axis.legend(handles=_legend() + [
        Line2D([0], [0], color=MUTED, lw=1.2, linestyle="--",
               label="layer completion")], ncol=3, loc="upper right")
    _style_axis(timeline_axis)

    families = sorted(family_work, key=lambda name: sum(family_work[name]), reverse=True)
    y = np.arange(len(families))
    local = np.array([family_work[name][0] for name in families])
    overlap = np.array([family_work[name][1] for name in families])
    network = np.array([family_work[name][2] for name in families])
    work_axis.barh(y, local, color=COLORS["compute"], label="exposed local")
    work_axis.barh(y, overlap, left=local, color=COLORS["input_output"],
                   label="local/network overlap")
    work_axis.barh(y, network, left=local + overlap,
                   color=COLORS["communication"], hatch="//",
                   label="exposed network")
    work_axis.set_yticks(y)
    work_axis.set_yticklabels(families, fontsize=8)
    work_axis.invert_yaxis()
    work_axis.set_xlabel("summed logical-event work (μs), not layer wall latency")
    work_axis.set_ylabel("operation family")
    work_axis.legend(ncol=3, loc="lower right")
    _style_axis(work_axis)
    figure.tight_layout()
    _save(figure, png_path, pdf_path)


def render_layer_analysis(report, png_path: Path, pdf_path: Path) -> None:
    """Render layer/rank busy heatmap and local/network work composition."""
    layer_ids = [record["layer_id"] for record in report.layers]
    devices = sorted({item["device"] for item in report.slices})
    row_by_layer = {layer_id: index for index, layer_id in enumerate(layer_ids)}
    col_by_device = {device: index for index, device in enumerate(devices)}
    matrix = np.zeros((len(layer_ids), len(devices)))
    intervals = defaultdict(list)
    for item in report.slices:
        if item["layer_id"] is None:
            continue
        intervals[(item["layer_id"], item["device"])].append(
            (item["start_us"], item["end_us"]))
    from rooflang.programs.experiments.performance_report import _interval_union
    for (layer_id, device), values in intervals.items():
        matrix[row_by_layer[layer_id], col_by_device[device]] = _interval_union(values)

    figure, (heat_axis, work_axis) = plt.subplots(
        1, 2, figsize=(17, max(8, len(layer_ids) * 0.25 + 2)),
        gridspec_kw={"width_ratios": [1.0, 1.25]})
    image = heat_axis.imshow(matrix, aspect="auto", cmap="Blues", interpolation="nearest")
    heat_axis.set_yticks(range(len(layer_ids)))
    heat_axis.set_yticklabels([f"L{layer_id}" for layer_id in layer_ids], fontsize=7)
    heat_axis.set_xticks(range(len(devices)))
    heat_axis.set_xticklabels(devices, rotation=45, ha="right", fontsize=7)
    heat_axis.set_title("Layer × GPU busy time")
    heat_axis.set_xlabel("device")
    heat_axis.set_ylabel("layer")
    figure.colorbar(image, ax=heat_axis, label="busy interval union (μs)")

    local = np.array([record["exposed_local_work_us"] for record in report.layers])
    overlap = np.array([record["overlap_work_us"] for record in report.layers])
    network = np.array([record["exposed_network_work_us"] for record in report.layers])
    y = np.arange(len(layer_ids))
    work_axis.barh(y, local, color=COLORS["compute"], label="exposed local")
    work_axis.barh(y, overlap, left=local, color=COLORS["input_output"],
                   label="local/network overlap")
    work_axis.barh(y, network, left=local + overlap,
                   color=COLORS["communication"], hatch="//",
                   label="exposed network")
    work_axis.set_yticks(y)
    work_axis.set_yticklabels([f"L{layer_id}" for layer_id in layer_ids], fontsize=7)
    work_axis.invert_yaxis()
    work_axis.set_xlabel("summed logical-event work (μs), not layer wall latency")
    work_axis.set_title("Per-layer local / overlap / network work")
    work_axis.legend(loc="lower right")
    _style_axis(work_axis)
    figure.tight_layout()
    _save(figure, png_path, pdf_path)
