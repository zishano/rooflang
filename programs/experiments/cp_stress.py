"""Bounded CP diagnostics using the unchanged legacy simulator and validator.

64M is a synthetic capacity stress, not a claim of model context support.
Resume requires matching source hashes and the exact experimental plan.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path

from rooflang.programs.experiments import find_pareto_frontier as pareto
from rooflang.programs.models import load_model
from rooflang.programs.models.dsv4_flash.optimization import _validate_args


def make_plan(contexts, stress_context):
    plan = []
    for context in contexts:
        # Invalid requests are retained as evidence; never silently round batch.
        for batch, pp, purpose in (
            (1, 1, 'single_request'), (1, 32, 'single_request_max_pp'),
            (8, 1, 'small_batch_no_pipeline'),
            (2, 32, 'minimum_legal_batch'),
            (8, 8, 'small_batch_pipeline'), (64, 1, 'full_batch_control'),
        ):
            if purpose == 'full_batch_control':
                continue  # Existing H200 Pareto data covers the 64K full-batch point.
            for cp in (1, 2):
                plan.append(dict(context=context, batch=batch, pp=pp, cp=cp,
                                 purpose=purpose, synthetic=False))
    for cp in (1, 2):
        plan.append(dict(context=stress_context, batch=2, pp=32, cp=cp,
                         purpose='synthetic_kv_capacity', synthetic=True))
    return plan


def case_for(spec):
    ep = 64 // spec['pp']
    return pareto.Case(
        workload=f"decode-{spec['context']}-tokens", hardware='h200',
        n_gpus=64, batch_size=spec['batch'], cp=spec['cp'], dp=ep // spec['cp'],
        ep=ep, pp_partition=pareto._balanced_partition(43, spec['pp']))


def run_one(spec):
    pareto._select_model('dsv4_flash')
    case = case_for(spec)
    try:
        _validate_args([None] * 43, case.batch_size, spec['context'], False,
                       case.cp, case.dp, case.ep, case.pp_partition, 64)
    except ValueError as exc:
        return {**asdict(case), **spec, 'case_id': case.case_id,
                'status': 'invalid', 'error': str(exc)}
    # This process-local workload entry changes only the requested sequence
    # length, not model semantics, timing formulas, or feasibility thresholds.
    pareto.WORKLOADS[case.workload] = ('decode', spec['context'])
    result = pareto.run_case(case)
    result.update(spec)
    return result


def capacity_probe(context):
    model = load_model('dsv4_flash')
    _, layers, _, _, reads, _ = model.declare_model(
        batch_size=1, seq_prefill=context, decode=True)
    main = sum(float(t.size_bytes) for k in reads for t in k.outputs.values())
    index = sum(float(t.size_bytes) for l in layers
                if l.index_cache_read is not None
                for t in l.index_cache_read.outputs.values())
    return dict(context=context, batch=1, main_kv_gb=main / 1e9,
                index_cache_gb=index / 1e9, combined_gb=(main + index) / 1e9,
                hbm_capacity_gb=144, kv_alone_exceeds_one_hbm=main + index > 144e9,
                note='Declared full-model cache for one request, excluding weights/buffers; not a runtime peak or single-request simulation.')


def write_csv(path, rows):
    columns = sorted(set().union(*(r.keys() for r in rows)))
    with path.open('w') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                         for k, v in r.items()} for r in rows)


def report(out, records, probes):
    records = sorted(records, key=lambda r: (r['context'], r['pp'], r['batch'], r['cp']))
    write_csv(out / 'all_points.csv', records)
    pairs = []
    baselines = {(r['context'], r['batch'], r['pp']): r
                 for r in records if r['cp'] == 1}
    for row in records:
        if row['cp'] != 2:
            continue
        base = baselines[(row['context'], row['batch'], row['pp'])]
        pair = dict(context=row['context'], batch=row['batch'], pp=row['pp'],
                    synthetic=row['synthetic'], cp1_status=base['status'],
                    cp2_status=row['status'])
        for cp, record in ((1, base), (2, row)):
            for field in ('latency_ms', 'preload_ms', 'tokens_per_s_user_elapsed',
                          'tokens_per_s_user_overlapped', 'tokens_per_s_gpu_overlapped',
                          'kv_cache_hbm_gb', 'peak_hbm_gb', 'peak_hbm_gb_overlapped',
                          'memory_feasible_overlapped', 'error'):
                pair[f'cp{cp}_{field}'] = record.get(field)
        if base['status'] == row['status'] == 'ok':
            pair['latency_delta_percent'] = (row['latency_ms'] / base['latency_ms'] - 1) * 100
            pair['overlapped_user_delta_percent'] = (
                row['tokens_per_s_user_overlapped'] / base['tokens_per_s_user_overlapped'] - 1) * 100
        pairs.append(pair)
    write_csv(out / 'paired_comparison.csv', pairs)
    (out / 'capacity_probe.json').write_text(json.dumps(probes, indent=2))
    lines = ['# Small-batch and long-context CP diagnostic', '',
             'Model: dsv4_flash; hardware: 64 H200 GPUs. CP1/CP2 pairs fix batch, context and PP. Original model/validator/accounting are unchanged. This is simulation, not hardware measurement.', '',
             '## Coverage and limitations', '',
             '- Single request on 64 GPUs is unsupported: batch must be divisible by CP*DP=64/PP, and PP cannot exceed the 43 layers (largest legal divisor is 32).',
             '- Small batches 2 and 8 require PP32 and PP8 here. These test pipelines, not PP1 low-batch CP acceleration. Never compare different PP as evidence of a CP-only gain.',
             '- 64M (or an explicitly supplied stress length) is a synthetic extrapolation for capacity diagnosis; it does not establish supported model context length.',
             '- Raw simulator OOM and projected overlapped memory infeasibility are reported separately. Preload is outside decode latency. Legacy sparse top-k/softmax merge and read-only KV modeling limitations remain.', '',
             '## Pairwise results', '',
             '| Context | Batch | PP | CP1 status | CP2 status | CP1 latency ms | CP2 latency ms | CP2 overlapped user delta % |',
             '|---:|---:|---:|---|---|---:|---:|---:|']
    def fmt(value):
        return '-' if value is None else f'{value:.6f}'
    for p in pairs:
        lines.append(f"| {p['context']} | {p['batch']} | {p['pp']} | {p['cp1_status']} | {p['cp2_status']} | {fmt(p['cp1_latency_ms'])} | {fmt(p['cp2_latency_ms'])} | {fmt(p.get('overlapped_user_delta_percent'))} |")
    lines += ['', '## Single-request cache size (declaration only)', '',
              '| Context | Main KV GB | Index GB | Combined GB | Exceeds 144 GB |',
              '|---:|---:|---:|---:|---|']
    for p in probes:
        lines.append(f"| {p['context']} | {p['main_kv_gb']:.6f} | {p['index_cache_gb']:.6f} | {p['combined_gb']:.6f} | {p['kv_alone_exceeds_one_hbm']} |")
    lines += ['', 'Each worker uses the existing run_case implementation. Raw JSONL preserves errors, OOM details and timing/memory metrics. No OOM-based skipping is performed. Rerun the shell launcher with the same output directory to resume; changed sources/plan require a new directory.']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--contexts', type=int, nargs='+', default=[1048576])
    parser.add_argument('--stress-context', type=int, default=67108864)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or min(args.contexts + [args.stress_context]) < 1:
        parser.error('workers and contexts must be positive')
    plan = make_plan(sorted(set(args.contexts)), args.stress_context)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    root = Path(__file__).resolve().parents[2]
    sources = sorted(set(root.glob('language/**/*.py')) | set(root.glob('runtime/**/*.py')) |
                     set(root.glob('programs/models/dsv4_flash/*.py')) |
                     {Path(__file__).resolve(), root / 'programs/experiments/find_pareto_frontier.py',
                      root / 'programs/presets/h200.py'})
    manifest = dict(model='dsv4_flash', hardware='h200', n_gpus=64, plan=plan,
                    source_hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sources})
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    mp = out / 'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise SystemExit('Source or plan changed: use a new --output-dir')
    mp.write_text(json.dumps(manifest, indent=2))
    raw = out / 'raw_results.jsonl'
    records = [json.loads(l) for l in raw.read_text().splitlines() if l.strip()] if raw.exists() else []
    done = {r['case_id'] for r in records}
    pending = [s for s in plan if case_for(s).case_id not in done]
    probes = []
    for context in sorted(set(args.contexts + [args.stress_context])):
        probe = capacity_probe(context)
        probes.append(probe)
        print(f"Cache probe: context={context}, one-request KV+index={probe['combined_gb']:.6f} GB", flush=True)
    report(out, records, probes)
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        tasks = {pool.submit(run_one, s): s for s in pending}
        for future in as_completed(tasks):
            record = future.result()
            with raw.open('a') as file:
                file.write(json.dumps(record) + '\n')
                file.flush()
            records.append(record)
            print(f"[{len(records)}/{len(plan)}] {record['case_id']} {record['status']} "
                  f"latency_ms={record.get('latency_ms')} error={record.get('error', '')}", flush=True)
    report(out, records, probes)
    print(f'Report: {out / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
