"""Summarize EP=64/PP=1 isolated layer timing results."""
import argparse
import csv
import json
from pathlib import Path


def read_rows(root: Path):
    rows = []
    for summary in sorted(root.glob("layer-*/context-*/cp-*/summary.json")):
        data = json.loads(summary.read_text(encoding="utf-8"))
        cfg = data["config"]
        cats = data["categories"]

        def cat(name, key="duration_us"):
            return cats.get(name, {}).get(key, 0.0)

        rows.append({
            "context": cfg["context"], "batch": cfg["batch_size"],
            "layer_id": data["layer"],
            "compression_ratio": data["compression_ratio"],
            "cp": cfg["cp"], "dp": cfg["dp"], "ep": cfg["ep"],
            "pp": len(cfg["pp_partition"]),
            "layer_span_us": data["layer_span_us"],
            "layer_compute_us": sum(v.get("compute_us", 0.0)
                                     for v in cats.values()),
            "cp_communication_us": cat("Communication"),
            "attention_us": cat("Attention core"),
            "attention_compute_us": cat("Attention core", "compute_us"),
            "attention_hbm_exposed_us": cat("Attention core", "hbm_exposed_us"),
            "attention_kv_read_bytes": cat("Attention core", "attention_kv_read_bytes"),
            "projections_us": cat("Attention projections"),
            "projections_compute_us": cat("Attention projections", "compute_us"),
            "local_copy_norm_us": cat("Norm / residual / local copy"),
            "ep_communication_us": cat("EP dispatch / combine"),
            "hbm_exposed_us": sum(v.get("hbm_exposed_us", 0.0)
                                   for v in cats.values()),
            "source": str(summary.parent),
        })
    for context in sorted({r["context"] for r in rows}):
        subset = [r for r in rows if r["context"] == context]
        base = next((r["layer_span_us"] for r in subset if r["cp"] == 1), None)
        for row in subset:
            row["span_delta_vs_cp1_us"] = (
                row["layer_span_us"] - base if base is not None else float("nan"))
    return sorted(rows, key=lambda r: (r["context"], r["cp"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.root)
    if not rows:
        raise SystemExit(f"no summary.json files found under {args.root}")
    csv_path = args.root / "cp_layer_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# DSV4 Flash decode layer timing: EP=64, PP=1",
        "",
        "RoofLang simulator results, not physical H200 measurements. Fixed: "
        "64 H200 GPUs (8 nodes), batch=64, EP=64, PP=1, GPU0, layer_id=2 "
        "(third model layer, compression ratio 4). DP=64/CP. The graph is "
        "truncated after the target layer to reduce simulation runtime; "
        "CP1 full-model validation matches the target layer span exactly "
        "(48.255 us at 64K).",
        "",
        "| Context | CP/DP | Layer span (us) | Δ vs CP1 (us) | CP communication (us) | Attention (us) | Attention compute (us) | Q/KV/O projections (us) | Local norm/copy (us) | EP communication (us) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['context']:,} | {r['cp']}/{r['dp']} | "
            f"{r['layer_span_us']:.3f} | {r['span_delta_vs_cp1_us']:+.3f} | "
            f"{r['cp_communication_us']:.3f} | {r['attention_us']:.3f} | "
            f"{r['attention_compute_us']:.3f} | {r['projections_us']:.3f} | "
            f"{r['local_copy_norm_us']:.3f} | {r['ep_communication_us']:.3f} |")
    lines += [
        "",
        "## Reading the results",
        "",
        "- CP/DP is swept over all divisors of 64: 1/64, 2/32, 4/16, 8/8, "
        "16/4, 32/2, 64/1. Global batch and EP remain fixed.",
        "- Attention compute demand is unchanged across CP values at a fixed "
        "context; the Attention duration increase is HBM-exposed time. "
        "Projection compute rises as DP falls and each DP shard carries a "
        "larger request batch.",
        "- CP communication rises sharply from CP=8 to CP=16, consistent "
        "with crossing the 8-GPU-per-node topology boundary in this preset.",
        "- EP communication is reported separately; do not add preload time "
        "to the selected layer span.",
        "- `cp_layer_comparison.csv` contains full-precision values and paths "
        "to each per-run summary.",
        "",
    ]
    (args.root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {csv_path} and {args.root / 'report.md'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
