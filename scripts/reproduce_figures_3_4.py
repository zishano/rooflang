"""Reproduce the Figure 3/4 experiment layout using isolated source snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
MODELS = ['dsv4_flash', 'glm_5_2_fp8', 'dsv4_pro', 'kimi_k3']
HARDWARE = ['h200', 'gh200', 'b300', 'gb300']
FLASH_PAPER_CONFIG = dict(models=['dsv4_flash'], stages=['prefill', 'decode'],
                          contexts=['64k', '1m'], hardware=HARDWARE,
                          gpu_counts=[8, 64], pp_degrees=None, max_batch_size=None)
LABELS = {'dsv4_flash': 'DeepSeek V4 Flash', 'dsv4_pro': 'DeepSeek V4 Pro',
          'glm_5_2_fp8': 'GLM-5.2-FP8 (repository model)', 'kimi_k3': 'Kimi K3'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(destination, source):
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    package = destination / 'source' / 'rooflang'
    package.mkdir(parents=True)
    if source == 'head':
        paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit],
                                        cwd=ROOT, text=True).splitlines()
    else:
        paths = [str(p.relative_to(ROOT)) for folder in ('programs', 'runtime', 'language')
                 for p in (ROOT / folder).rglob('*.py')] + ['__init__.py']
    for name in paths:
        if not name.endswith('.py') or not (name == '__init__.py' or
                name.split('/')[0] in ('programs', 'runtime', 'language')):
            continue
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
                           if source == 'head' else (ROOT / name).read_bytes())
    # Pin this independent driver as well, including when resuming in a later session.
    (destination / 'driver.py').write_bytes(Path(__file__).read_bytes())
    hashes = {str(p.relative_to(destination)): digest(p) for p in package.rglob('*.py')}
    hashes['driver.py'] = digest(destination / 'driver.py')
    return commit, hashes


def commands(out, config, workers):
    workloads = [f'{stage}-{context}' for stage in config['stages'] for context in config['contexts']]
    for model in config['models']:
        cmd = [sys.executable, '-B', '-m', 'rooflang.programs.experiments.find_pareto_frontier',
               '--model', model, '--workloads', *workloads, '--hardware', *config['hardware'],
               '--gpu-counts', *map(str, config['gpu_counts']), '--workers', str(workers),
               '--no-plot', '--output-dir', str(out / model)]
        if config['max_batch_size'] is not None:
            cmd += ['--max-batch-size', str(config['max_batch_size'])]
        if config['pp_degrees']:
            cmd += ['--pp-degrees', *map(str, config['pp_degrees'])]
        yield cmd


def plot(out, manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import EngFormatter
    from rooflang.programs.experiments.find_pareto_frontier import pareto_frontier, _write_csv

    cfg = manifest['config']
    all_points, selected = [], []
    for model in cfg['models']:
        raw = out / model / 'raw_results.jsonl'
        if not raw.exists():
            continue
        records = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
        latest = {r['case_id']: dict(r, model=model) for r in records}
        all_points.extend(latest.values())
    colors = dict(zip(HARDWARE, ['#0072B2', '#E69F00', '#009E73', '#CC79A7']))
    for stage in cfg['stages']:
        ncols = len(cfg['contexts']) * len(cfg['gpu_counts'])
        fig, axes = plt.subplots(len(cfg['models']), ncols, squeeze=False,
                                 figsize=(max(6, 4 * ncols), 3.2 * len(cfg['models']) + 1.2))
        for row, model in enumerate(cfg['models']):
            col = 0
            for context in cfg['contexts']:
                for gpus in cfg['gpu_counts']:
                    ax = axes[row, col]
                    count = 0
                    for hw in cfg['hardware']:
                        records = [r for r in all_points if r['model'] == model and
                                   r['workload'] == f'{stage}-{context}' and
                                   r['n_gpus'] == gpus and r['hardware'] == hw]
                        frontier = pareto_frontier(records, 'tokens_per_s_user_overlapped',
                                                   'tokens_per_s_gpu_overlapped',
                                                   'memory_feasible_overlapped')
                        selected.extend(frontier)
                        count += len(frontier)
                        ax.plot([r['tokens_per_s_user_overlapped'] for r in frontier],
                                [r['tokens_per_s_gpu_overlapped'] for r in frontier],
                                '.-', color=colors[hw], label=hw.upper(), markersize=3)
                    if not count:
                        ax.text(.5, .5, 'No feasible completed points', transform=ax.transAxes,
                                ha='center', fontsize=8)
                    ax.set_title(f'{LABELS[model]}\n{context} {stage}, {gpus} GPUs', fontsize=10)
                    ax.set(xlabel='tokens/s/user', ylabel='tokens/s/GPU', xlim=(0, None), ylim=(0, None))
                    ax.xaxis.set_major_formatter(EngFormatter(sep=''))
                    ax.yaxis.set_major_formatter(EngFormatter(sep=''))
                    ax.grid(alpha=.2)
                    col += 1
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4)
        number = 3 if stage == 'prefill' else 4
        fig.suptitle(f'Figure {number} reproduction\nIdeal overlap, repository models', fontsize=12)
        fig.tight_layout(rect=(0, .10, 1, .90))
        for suffix in ('png', 'pdf'):
            fig.savefig(out / f'figure_{number}_{stage}.{suffix}', dpi=180)
        plt.close(fig)
    _write_csv(out / 'all_points.csv', all_points)
    _write_csv(out / 'paper_overlapped_frontiers.csv', selected)
    status = {key: sum(r.get('status') == key for r in all_points)
              for key in sorted({r.get('status', 'unknown') for r in all_points})}
    (out / 'summary.json').write_text(json.dumps({'status_counts': status,
        'frontier_points': len(selected), 'search_finished': manifest.get('search_finished', False),
        'bounded_batch_search': cfg['max_batch_size'] is not None,
        'note': ('GLM repository implementation is not verified equivalent to paper GLM-5.3; '
                 if 'glm_5_2_fp8' in cfg['models'] else '') +
                'HEAD is a reproducible baseline, not a verified paper release.'}, indent=2) + '\n')
    print(f'Plots and tables: {out}', flush=True)


def internal(out, mode, workers):
    manifest = json.loads((out / 'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if digest(out / name) != expected:
            raise RuntimeError(f'Snapshot changed: {name}')
    if mode != 'plot':
        for cmd in commands(out, manifest['config'], workers):
            if mode == 'dry':
                cmd += ['--dry-run']
            print(' '.join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        if mode == 'run':
            manifest['search_finished'] = True
            (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if mode != 'dry':
        plot(out, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--source', choices=['head', 'workspace'], default='head')
    parser.add_argument('--flash-paper', action='store_true',
                        help='Fix the complete Flash-only Figure 3/4 experiment matrix')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--stages', nargs='+', choices=['prefill', 'decode'], default=['prefill', 'decode'])
    parser.add_argument('--contexts', nargs='+', choices=['64k', '1m'], default=['64k', '1m'])
    parser.add_argument('--hardware', nargs='+', choices=HARDWARE, default=HARDWARE)
    parser.add_argument('--gpu-counts', nargs='+', type=int, choices=[8, 64], default=[8, 64])
    parser.add_argument('--pp-degrees', nargs='+', type=int)
    parser.add_argument('--max-batch-size', type=int)
    parser.add_argument('--workers', type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--plot-only', action='store_true')
    parser.add_argument('--internal', choices=['dry', 'run', 'plot'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 1 or args.max_batch_size is not None and args.max_batch_size < 1:
        parser.error('workers and max-batch-size must be positive')
    if args.pp_degrees and any(p <= 0 for p in args.pp_degrees):
        parser.error('PP degrees must be positive')
    if args.internal:
        internal(args.output_dir.resolve(), args.internal, args.workers)
        return
    if args.flash_paper:
        for key, value in FLASH_PAPER_CONFIG.items():
            setattr(args, key, value)
    temporary = tempfile.TemporaryDirectory(prefix='rooflang-fig34-') if args.dry_run else None
    if (args.resume or args.plot_only) and args.output_dir is None:
        parser.error('--resume/--plot-only require --output-dir')
    out = (Path(temporary.name) / 'run' if temporary else args.output_dir or
           ROOT / 'results' / ('paper_figures_3_4_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))).resolve()
    if args.resume or args.plot_only:
        if not (out / 'manifest.json').is_file():
            parser.error('No manifest.json in the specified output directory')
        saved = json.loads((out / 'manifest.json').read_text())
        if args.flash_paper and saved['config'] != FLASH_PAPER_CONFIG:
            parser.error('Saved experiment is not the complete Flash-only paper matrix')
        print('Using saved snapshot and experiment configuration; selection flags are ignored.', flush=True)
    else:
        out.mkdir(parents=True, exist_ok=False)
        commit, hashes = snapshot(out, args.source)
        cfg = {key: getattr(args, key) for key in ['models', 'stages', 'contexts', 'hardware',
                                                  'gpu_counts', 'pp_degrees', 'max_batch_size']}
        manifest = {'config': cfg, 'source': args.source, 'git_commit': commit, 'sha256': hashes,
                    'search_finished': False, 'metric': 'overlapped only',
                    'paper': '2609.12551v2, Section 3.1, Figures 3/4, Tables 5/6'}
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    env = dict(os.environ, PYTHONPATH=str(out / 'source'), MPLBACKEND='Agg',
               MPLCONFIGDIR=str(out / '.matplotlib'))
    print(f'Source snapshot/results: {out}', flush=True)
    try:
        subprocess.run([sys.executable, '-B', str(out / 'driver.py'), '--internal',
                        'dry' if args.dry_run else 'plot' if args.plot_only else 'run',
                        '--output-dir', str(out), '--workers', str(args.workers)],
                       cwd=out / 'source', env=env, check=True)
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == '__main__':
    main()
