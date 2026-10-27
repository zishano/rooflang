#!/usr/bin/env python3
"""Summarize CP1/2/4 single-layer timing runs into a comparison table."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True,
                        help='Experiment root containing layer-2/context-*/cp-*')
    parser.add_argument('--layer', type=int, default=2)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = []
    for context in (65536, 1048576):
        group = []
        for cp in (1, 2, 4):
            folder = root / f'layer-{args.layer}' / f'context-{context}' / f'cp-{cp}'
            summary = json.loads((folder / 'summary.json').read_text())
            categories = summary['categories']
            cp_comm = sum(categories.get(name, {}).get('network_elapsed_us', 0.0)
                          for name in ('Communication', 'Layer output gather'))
            ep_comm = categories.get('EP dispatch / combine', {}).get(
                'network_elapsed_us', 0.0)
            row = {
                'context': context,
                'batch_size': summary['config']['batch_size'],
                'layer': args.layer,
                'compression_ratio': summary['compression_ratio'],
                'cp': cp,
                'dp': summary['config']['dp'],
                'ep': summary['config']['ep'],
                'pp': len(summary['config']['pp_partition']),
                'gpu_index': summary['config']['gpu_index'],
                'layer_span_us': summary['layer_span_us'],
                'layer_compute_us': summary['totals']['compute_us'],
                'layer_hbm_exposed_us': summary['totals']['hbm_exposed_us'],
                'attention_core_us': categories.get('Attention core', {}).get(
                    'duration_us', 0.0),
                'cp_network_us': cp_comm,
                'ep_network_us': ep_comm,
                'total_network_us': cp_comm + ep_comm,
                'full_model_kv_preload_us': summary['full_model_kv_preload_us'],
                'full_decode_measured_us': summary['full_run_us'],
                'event_count': summary['event_count'],
                'source_dir': str(folder),
            }
            group.append(row)
        baseline = group[0]['layer_span_us']
        for row in group:
            row['layer_span_delta_vs_cp1_percent'] = (
                row['layer_span_us'] / baseline - 1) * 100
            rows.append(row)

    out_csv = root / 'cp_layer_comparison.csv'
    with out_csv.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        '# DSV4 Flash CP single-layer comparison', '',
        'Simulation only: H200 preset, 64 GPUs, B=64, EP=8, PP=8, layer '
        f'{args.layer} (compression ratio 4), GPU0. CP1/2/4 use DP8/4/2. '
        'The layer span is the selected GPU layer window; KV preload is '
        'reported separately and excluded from this span.', '',
        '| Context | CP/DP | Layer span (us) | Δ vs CP1 | Attention core (us) | '
        'CP network (us) | EP network (us) | Full KV preload (us) |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['context']} | {row['cp']}/{row['dp']} | "
            f"{row['layer_span_us']:.3f} | "
            f"{row['layer_span_delta_vs_cp1_percent']:+.2f}% | "
            f"{row['attention_core_us']:.3f} | {row['cp_network_us']:.3f} | "
            f"{row['ep_network_us']:.3f} | "
            f"{row['full_model_kv_preload_us']:.3f} |"
        )
    lines += [
        '',
        'CP network sums the events categorized as Communication and Layer '
        'output gather on GPU0. EP dispatch/combine is shown separately. '
        'These are trace-attributed timings, not hardware measurements. '
        'The current analyzer starts timing at read_input, so KV preload is '
        'not part of layer_span_us.',
    ]
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    print(out_csv)
    print(root / 'report.md')


if __name__ == '__main__':
    main()
