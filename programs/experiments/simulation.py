# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""DeepSeek V4 Pro inference — Simulation Phase."""

from rooflang.runtime.simulator import Simulator
from rooflang.runtime.trace_export import export_trace


def simulate(
    g, p, hw, trace_path="dsv4_pro_prefill.json", measurement_start=None,
):
    """Run DES simulator and export trace."""
    result = Simulator(
        g, p, hw, measurement_start=measurement_start).run()
    if hasattr(g, "dcp_step_markers"):
        starts, ends = {}, {}
        markers = {kernel for pair in g.dcp_step_markers for kernel in pair}
        for entry in result.trace:
            if entry.kernel in markers:
                starts[entry.kernel] = min(starts.get(entry.kernel, float("inf")), entry.start_us)
                ends[entry.kernel] = max(ends.get(entry.kernel, 0.0), entry.end_us)
        result.dcp_metadata = {
            **getattr(p, "metadata", {}),
            "step_times_us": [ends[end] - starts[start] for start, end in g.dcp_step_markers],
        }
    export_trace(result, trace_path)
    return result
