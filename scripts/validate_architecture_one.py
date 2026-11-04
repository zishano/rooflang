"""Independent, serial DSV4 Flash PP-partition pilot; never resumes/overwrites."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from rooflang.programs.experiments import find_pareto_frontier as search
from rooflang.programs.models.dsv4_flash.optimization import _validate_args

PARTITIONS = ((22, 21), (21, 22), (23, 20), (20, 23), (24, 19))
MODEL = 'dsv4_flash'
METRICS = (
    'latency_ms', 'tokens_per_s_user', 'tokens_per_s_gpu',
    'tokens_per_s_user_elapsed', 'tokens_per_s_gpu_elapsed',
    'tokens_per_s_user_overlapped', 'tokens_per_s_gpu_overlapped',
    'peak_hbm_gb', 'peak_hbm_gb_elapsed', 'compute_ratio',
    'communication_ratio', 'cycle_ms', 'bubble_ratio',
)


def validated_cases():
    """Reuse enumeration, then validate every explicit non-balanced partition."""
    search._select_model(MODEL)
    candidates = list(search.enumerate_cases(
        ['decode-8k'], ['h200'], [8], [2], pp_degrees=[2], max_batch_size=8))
    matches = [c for c in candidates if (c.cp, c.dp, c.ep, c.batch_size) == (1, 4, 4, 8)]
    if len(matches) != 1 or matches[0].pp_partition != PARTITIONS[0]:
        raise ValueError('Expected exactly one enumerated [22,21] baseline')
    cases = [replace(matches[0], pp_partition=p) for p in PARTITIONS]
    for case in cases:
        if len(case.pp_partition) != 2 or any(n <= 0 for n in case.pp_partition):
            raise ValueError('Expected two positive stage layer counts')
        _validate_args(range(search.N_LAYERS), case.batch_size, 8192, False,
                       case.cp, case.dp, case.ep, case.pp_partition, case.n_gpus)
    return cases


def simulate(case):
    # spawn does not inherit the searcher's selected-model globals.
    search._select_model(MODEL)
    return search.run_case(case)


def comparison_rows(records):
    rows = []
    for record in records:
        row = dict(record)
        row['pp_partition'] = json.dumps(record['pp_partition'])
        if record['status'] == 'ok':
            row['cycle_ms'] = record['total_gpu_elapsed_ms'] / record['n_gpus']
            row['bubble_ratio'] = 1 - record['compute_ratio'] - record['communication_ratio']
        rows.append(row)
    baseline = rows[0] if rows and rows[0]['status'] == 'ok' else {}
    for row in rows:
        for metric in METRICS:
            base = baseline.get(metric)
            row[metric + '_delta_pct'] = (
                100 * (row[metric] / base - 1)
                if row['status'] == 'ok' and base and metric in row else None)
    return rows


def write_comparison(output, records):
    rows = comparison_rows(records)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / 'comparison.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        '# 架构一：独立分层仿真对照', '',
        'DSV4 Flash / H200 × 8 / decode-8k / batch=8 / CP=1 DP=4 EP=4 PP=2。',
        '基准为 [22,21]；百分比变化为 (当前/基准−1)×100。非实测 GPU 性能。', '',
        '| 分层 | 状态 | TPOT ms | 稳态周期 ms | 原始 tok/s/GPU | 稳态 tok/s/GPU | 理想重叠 tok/s/GPU | 稳态变化 % | HBM GB（稳态投影） | 稳态内存可行 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    keys = ('latency_ms', 'cycle_ms', 'tokens_per_s_gpu',
            'tokens_per_s_gpu_elapsed', 'tokens_per_s_gpu_overlapped',
            'tokens_per_s_gpu_elapsed_delta_pct', 'peak_hbm_gb_elapsed')
    for row in rows:
        values = [f'{row[k]:.6f}' if row.get(k) is not None else '—' for k in keys]
        lines.append('| ' + ' | '.join([row['pp_partition'], row['status'], *values,
                                       str(row.get('memory_feasible_elapsed', '—'))]) + ' |')
        if row.get('error'):
            lines.extend(['', f"错误 {row['pp_partition']}: {row['error']}", ''])
    lines.extend(['', '完整 user/GPU 三种吞吐、计算/通信/bubble 比例及相对变化见 comparison.csv。',
                  '稳态与理想重叠投影必须同时检查对应 memory_feasible 字段；不可行点不能作为可用性能。',
                  'cycle_ms 为原搜索器 total_gpu_elapsed_ms / 8；不提供未由 run_case 返回的单阶段实测时间。'])
    with (output / 'comparison.md').open('x') as stream:
        stream.write('\n'.join(lines) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Validate and print/write plan, without simulation')
    parser.add_argument('--output-dir', type=Path, help='Must not exist; default is a unique results/architecture_one directory')
    parser.add_argument('--limit', type=int, choices=range(1, 6), default=5, help='Run first N cases, always including baseline')
    args = parser.parse_args(argv)
    cases = validated_cases()  # Validate all five BEFORE creating outputs or launching workers.
    plan = {
        'model': MODEL, 'simulation_only': True, 'serial': True,
        'validated_case_count': len(cases), 'selected_case_count': args.limit,
        'dry_run': args.dry_run,
        'searcher_sha256': hashlib.sha256(Path(search.__file__).read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cases': [{'case_id': c.case_id, **asdict(c)} for c in cases],
    }
    if args.dry_run and args.output_dir is None:
        print(json.dumps(plan, indent=2))
        return 0
    output = args.output_dir or (Path(__file__).resolve().parents[2] / 'results' / 'architecture_one' /
                                datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'plan.json').open('x') as stream:
        json.dump(plan, stream, indent=2)
    print(f'Validated all five configurations. Output: {output.resolve()}', flush=True)
    if args.dry_run:
        return 0
    records = []
    with (output / 'raw_results.jsonl').open('x') as stream:
        # Exactly one worker, recycled after each case to release simulation state.
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                                 max_tasks_per_child=1) as pool:
            for case in cases[:args.limit]:
                print(f'Running {case.case_id}', flush=True)
                try:
                    record = pool.submit(simulate, case).result()
                except Exception as error:
                    record = {**asdict(case), 'case_id': case.case_id, 'status': 'error',
                              'error': f'{type(error).__name__}: {error}'}
                record['model'] = MODEL
                stream.write(json.dumps(record) + '\n')
                stream.flush()
                records.append(record)
                print(f"Finished {case.pp_partition}: {record['status']}", flush=True)
    write_comparison(output, records)
    return int(any(r['status'] != 'ok' for r in records))


if __name__ == '__main__':
    raise SystemExit(main())
