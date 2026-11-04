# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Model inference simulation entry point."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from rooflang.language.hardware.component import Compute
from rooflang.programs.models import MODEL_NAMES, load_model
from rooflang.programs.presets.b300 import B300Cluster, B300SuperChip
from rooflang.programs.presets.h200 import H200Cluster, H200SuperChip
from rooflang.programs.experiments.simulation import simulate


HARDWARE_MAP = {
    "B300Cluster1Node": lambda: B300Cluster(n_nodes=1),
    "B300SuperChip": lambda: B300SuperChip(),
    "H200Cluster1Node": lambda: H200Cluster(n_nodes=1),
    "H200SuperChip": lambda: H200SuperChip(),
}


def main():
    parser = argparse.ArgumentParser(description="Model inference simulation")
    parser.add_argument("--model", choices=MODEL_NAMES, default="dsv4_flash",
                        help="Model implementation (default: dsv4_pro)")
    parser.add_argument("--hardware", default="B300Cluster1Node",
                        choices=list(HARDWARE_MAP.keys()),
                        help="Hardware configuration")
    parser.add_argument("--stage", default="decode",
                        choices=("prefill", "decode"),
                        help="Inference stage to simulate")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size (default: config BATCH)")
    parser.add_argument("--context", type=int, default=8192,
                        help="Prefill length or decode KV context (default: 8192)")
    parser.add_argument("--cp", type=int, default=1,
                        help="Context-parallel degree (default: 1)")
    parser.add_argument("--dp", type=int, default=8,
                        help="Data-parallel degree (default: 8)")
    parser.add_argument("--ep", type=int, default=8,
                        help="Expert-parallel degree (default: 8)")
    parser.add_argument(
        "--pp-partition", type=int, nargs="+", default=[43],
        help="Layer counts assigned to successive pipeline stages")
    parser.add_argument(
        "--measurement_start", choices=("read_input", "none"),
        default="read_input",
        help="Kernel marking measurement start, or measure the full run "
             "(default: read_input)")
    parser.add_argument("--visualization", action="store_true",
                        help="Export layer graph visualization")
    parser.add_argument("--report", action="store_true",
                        help="Generate post-simulation timing analysis")
    report_paths = parser.add_mutually_exclusive_group()
    report_paths.add_argument(
        "--report-dir", type=Path,
        help="Exact report directory; it must not already exist")
    report_paths.add_argument(
        "--report-root", type=Path, default=Path("results/performance"),
        help="Parent for timestamped report directories")
    parser.add_argument(
        "--report-window", choices=("measured", "full"), default="measured",
        help="Analyze measured decode or the full run including preload")
    parser.add_argument("--focus-layer", type=int, default=0,
                        help="Layer used by the focused timeline (default: 0)")
    parser.add_argument(
        "--focus-device",
        help="Device name or trailing numeric rank for the focused timeline")
    parser.add_argument(
        "--keep-failed-report", action="store_true",
        help="Retain failed staging output under report-root/_failed")
    args = parser.parse_args()

    model = load_model(args.model)
    is_prefill = args.stage == "prefill"
    hw = HARDWARE_MAP[args.hardware]()

    # A. Declaration
    decl_kwargs = dict(
        seq_prefill=args.context,
        decode=not is_prefill)
    if args.batch_size is not None:
        decl_kwargs["batch_size"] = args.batch_size
    g, layers, emb, read_input, kv_cache_reads, output_head = \
        model.declare_model(**decl_kwargs)

    # B. Visualization
    if args.visualization:
        from rooflang.programs.experiments.visualization import visualize_layer

        viz_layer = None
        seeds = {emb, read_input}
        if layers:
            viz_layer = layers[0]
        if kv_cache_reads:
            seeds.add(kv_cache_reads[0])
        seeds.discard(None)
        if viz_layer:
            visualize_layer(g, viz_layer, extra_seeds=seeds)

    # C. Optimization
    if isinstance(hw, (B300SuperChip, H200SuperChip)):
        g, p = model.optimize_model_superchip(g, hw)
    else:
        n_gpus = sum(
            isinstance(component, Compute) and component.kind == "gpu"
            for component in hw.nodes)
        optimize_kwargs = dict(
            cp=args.cp, dp=args.dp, ep=args.ep,
            pp_partition=args.pp_partition or [model.N_LAYERS],
            n_gpus=n_gpus)
        if is_prefill:
            g, p = model.optimize_model_cluster_prefill(
                g, layers, hw, emb, read_input, output_head,
                **optimize_kwargs)
        else:
            g, p = model.optimize_model_cluster_decode(
                g, layers, hw, emb, read_input, kv_cache_reads,
                output_head, seq_prefill=decl_kwargs["seq_prefill"],
                **optimize_kwargs)

    # D. Simulation and optional performance report
    trace_name = f"{args.model}_{args.stage}_{args.hardware}.json"
    measurement_start = (
        read_input if args.measurement_start == "read_input" else None)
    result = simulate(
        g, p, hw, None if args.report else trace_name,
        measurement_start=measurement_start)
    duration_us = result.measured_time_us
    print(f"{args.stage} ({args.hardware}): "
          f"{duration_us:.1f} us ({duration_us / 1000:.1f} ms)")
    if result.measurement_start_us:
        print(f"KV preload: {result.measurement_start_us:.1f} us "
              f"({result.measurement_start_us / 1000:.1f} ms)")

    if args.report:
        from rooflang.programs.experiments.performance_report import (
            RunConfig, build_performance_report, publish_performance_report,
        )

        partition = tuple(args.pp_partition or [model.N_LAYERS])
        config = RunConfig(
            model=args.model,
            hardware=args.hardware,
            stage=args.stage,
            batch_size=args.batch_size,
            context=args.context,
            cp=args.cp,
            dp=args.dp,
            ep=args.ep,
            pp_partition=partition,
            measurement_start=args.measurement_start,
        )
        report = build_performance_report(
            result, g, p, layers, config=config, window=args.report_window)
        if args.report_dir is not None:
            report_dir = args.report_dir.resolve()
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            pp_name = "-".join(map(str, partition))
            run_name = (
                f"{args.model}_{args.stage}_{args.hardware}_b{args.batch_size}_"
                f"cp{args.cp}_dp{args.dp}_ep{args.ep}_pp{pp_name}_{stamp}")
            report_dir = args.report_root.resolve() / run_name
        repo_root = Path(__file__).resolve().parents[2]
        published = publish_performance_report(
            report, report_dir, repo_root=repo_root,
            focus_layer=args.focus_layer, focus_device=args.focus_device,
            keep_failed=args.keep_failed_report)
        print(f"Timing report: {published}")


if __name__ == "__main__":
    main()
