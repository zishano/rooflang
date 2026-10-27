# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Model inference simulation entry point."""

import argparse

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
    parser = argparse.ArgumentParser(
        description="Model inference simulation", allow_abbrev=False)
    parser.add_argument("--model", choices=MODEL_NAMES, default="dsv4_pro",
                        help="Model implementation (default: dsv4_pro)")
    parser.add_argument("--hardware", required=True,
                        choices=list(HARDWARE_MAP.keys()),
                        help="Hardware configuration")
    parser.add_argument("--stage", required=True,
                        choices=("prefill", "decode"),
                        help="Inference stage to simulate")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: config BATCH)")
    parser.add_argument('--context', type=int, default=8192,
                        help='prefill 序列长度或 decode KV context 长度')
    parser.add_argument("--cp", type=int, default=1,
                        help="Context-parallel degree (default: 1)")
    parser.add_argument("--dcp", action="store_true",
                        help="Enable DSV4 decode DCP and run its two-step numerical reference check")
    parser.add_argument("--decode-steps", type=int, default=None,
                        help="Stateful DCP decode steps (default: 1 with --dcp)")
    parser.add_argument("--dcp-trace", choices=("none", "first", "all"), default="none",
                        help="DCP per-step traces (default: compact summary only)")
    parser.add_argument("--dcp-trace-steps", type=int, nargs="+", default=None,
                        help="Export only these 1-based DCP steps, e.g. 3 4 7 8")
    parser.add_argument("--dp", type=int, default=8,
                        help="Data-parallel degree (default: 8)")
    parser.add_argument("--ep", type=int, default=8,
                        help="Expert-parallel degree (default: 8)")
    parser.add_argument(
        "--pp-partition", type=int, nargs="+",
        help="Layer counts assigned to successive pipeline stages")
    parser.add_argument(
        "--measurement_start", choices=("read_input", "none"),
        default="read_input",
        help="Kernel marking measurement start, or measure the full run "
             "(default: read_input)")
    parser.add_argument("--visualization", action="store_true",
                        help="Export layer graph visualization")
    args = parser.parse_args()

    if args.dcp and args.model != "dsv4_flash":
        parser.error("--dcp is only supported for --model dsv4_flash")
    if args.dcp and args.stage != "decode":
        parser.error("--dcp requires --stage decode")
    if args.dcp and args.hardware.endswith("SuperChip"):
        parser.error("--dcp requires a cluster hardware configuration")
    if args.dcp and (args.cp <= 0 or args.dp <= 0 or args.ep != args.cp * args.dp):
        parser.error("--dcp requires positive CP/DP and DP * CP == EP")
    if args.decode_steps is not None and (not args.dcp or args.decode_steps < 1):
        parser.error("--decode-steps requires --dcp and a positive step count")
    if args.dcp_trace_steps is not None and not args.dcp:
        parser.error("--dcp-trace-steps requires --dcp")
    if args.dcp_trace_steps is not None and args.dcp_trace != "none":
        parser.error("choose either --dcp-trace or --dcp-trace-steps")

    reference_check = None
    if args.dcp:
        from rooflang.programs.models.dsv4_flash.dcp_validation import validate_dcp_startup

        reference_check = validate_dcp_startup(args.cp)
        print(
            f"DCP numerical reference: PASS (CP={args.cp}, 2 steps, "
            f"max_abs_error={reference_check['max_abs_error']:.3e})",
            flush=True)
        print(
            "Validation scope: production DCP state/top-k/owner/LSE graph operators "
            "with small encoded tensors; learned compression and quantization "
            "are cost abstractions. Simulation: persistent multi-step decode.",
            flush=True)

    model = load_model(args.model)
    is_prefill = args.stage == "prefill"
    hw = HARDWARE_MAP[args.hardware]()

    # A. Declaration
    decl_kwargs = dict(
        seq_prefill=args.context,
        decode=not is_prefill)
    if args.batch_size is not None:
        decl_kwargs["batch_size"] = args.batch_size
    if args.dcp:
        from rooflang.programs.models.dsv4_flash.dcp_reuse import ReusableDCP

        n_gpus = sum(isinstance(component, Compute) and component.kind == "gpu"
                     for component in hw.nodes)
        plan = ReusableDCP(
            hw, batch_size=args.batch_size if args.batch_size is not None else model.BATCH,
            context=args.context, cp=args.cp, dp=args.dp, ep=args.ep,
            pp_partition=args.pp_partition or [model.N_LAYERS], n_gpus=n_gpus,
            steps=args.decode_steps or 1)
        g, p, read_input = plan.graph, plan.placement, plan.reader
        layers, emb, kv_cache_reads, output_head = [], None, [], ()
    else:
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
    if args.dcp:
        pass  # Stateful DCP builder completes graph transforms and placement.
    elif isinstance(hw, (B300SuperChip, H200SuperChip)):
        g, p = model.optimize_model_superchip(g, hw)
    else:
        n_gpus = sum(
            isinstance(component, Compute) and component.kind == "gpu"
            for component in hw.nodes)
        optimize_kwargs = dict(
            cp=args.cp, dp=args.dp, ep=args.ep,
            pp_partition=args.pp_partition or [model.N_LAYERS],
            n_gpus=n_gpus)
        if not is_prefill and args.model == "dsv4_flash":
            optimize_kwargs["dcp_enabled"] = args.dcp
        if is_prefill:
            g, p = model.optimize_model_cluster_prefill(
                g, layers, hw, emb, read_input, output_head,
                **optimize_kwargs)
        else:
            g, p = model.optimize_model_cluster_decode(
                g, layers, hw, emb, read_input, kv_cache_reads,
                output_head, seq_prefill=decl_kwargs["seq_prefill"],
                **optimize_kwargs)

    if not is_prefill and args.model == "dsv4_flash":
        if reference_check is not None:
            p.metadata["dcp_reference_validation"] = reference_check
        route = "DCP" if args.dcp else "legacy decode CP"
        print(f"Decode route: {route} (CP={args.cp}, DP={args.dp}, EP={args.ep}), PP={args.pp_partition}")

    # D. Simulation
    trace_name = f"{args.model}_{args.stage}_{args.hardware}.json"
    if args.dcp:
        trace_name = f"{args.model}_{args.stage}_{args.hardware}_dcp_{plan.steps}steps.json"
    measurement_start = (
        read_input if args.measurement_start == "read_input" else None)
    if args.dcp:
        from rooflang.programs.models.dsv4_flash.dcp_reuse import simulate_reused_decode

        def progress(step, duration, wall_seconds):
            print(f"decode step {step}: {duration:.3f} us "
                  f"(context={args.context + step - 1} -> {args.context + step}, "
                  f"wall={wall_seconds:.3f} s)", flush=True)

        result = simulate_reused_decode(plan, trace_name, trace_mode=args.dcp_trace,
                                        trace_steps=args.dcp_trace_steps, progress=progress)
    else:
        result = simulate(g, p, hw, trace_name, measurement_start=measurement_start)
    duration_us = (result.total_time_us if args.measurement_start == "none"
                   else result.measured_time_us)
    stage_label = f"decode {plan.steps} steps" if args.dcp else args.stage
    print(f"{stage_label} ({args.hardware}): "
          f"{duration_us:.1f} us ({duration_us / 1000:.1f} ms)")
    if args.dcp:
        print(f"decode total: {duration_us:.3f} us for {plan.steps} steps")
        meta = result.dcp_metadata
        print(f"Wall time: graph build={meta['wall_build_seconds']:.3f} s, "
              f"simulation={meta['wall_simulation_seconds']:.3f} s, "
              f"trace export={meta['wall_export_seconds']:.3f} s; graph built once")
        print(f"DCP summary: {trace_name} (trace mode: {args.dcp_trace})")
    if result.measurement_start_us:
        print(f"KV preload: {result.measurement_start_us:.1f} us "
              f"({result.measurement_start_us / 1000:.1f} ms)")


if __name__ == "__main__":
    main()
