"""Simulate Kimi K3 decode on an eight-GPU GB300 domain and report one layer.

Run with the repository parent on PYTHONPATH. No existing trace is overwritten.
"""
import argparse
import csv
import subprocess
import sys
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

from rooflang.programs.models import load_model
from rooflang.programs.presets.gb300 import GB300Cluster
from rooflang.language.hardware.component import Compute
from rooflang.runtime.simulator import Simulator
from rooflang.language.kernels.comm import CommKernel


OUT = Path(__file__).resolve().parents[1] / 'results/kimi_k3_gb300_8gpu_layer_timing'
DEVICE = 'n0-nvidia-gb300-0'
LAYER = 0


def category(field, kernel, entry):
    if field in ('dispatch', 'combine'):
        return 'EP dispatch / combine'
    if isinstance(kernel, CommKernel) and entry.network_time_us > 0:
        return 'Communication'
    if field in ('sa', 'kda_f_a', 'kda_f_b', 'kda_beta'):
        return 'Attention core'
    if field in (
        'wq_a', 'wq_b', 'wkv', 'wo_a', 'wo_b', 'wo',
        'kda_wq', 'kda_wk', 'kda_wv', 'attn_gate', 'output_gate',
    ):
        return 'Attention projections'
    if field.startswith('expert'):
        return 'MoE experts'
    if field in ('sw_up', 'sw_down'):
        return 'Shared expert'
    if field == 'gate':
        return 'MoE gate'
    if field in ('kv_sink', 'kv_cache_quant', 'state_store'):
        return 'KV cache write'
    return 'Norm / residual / local copy'


def parse_args():
    parser = argparse.ArgumentParser(description='重新模拟 prefill/decode，提取单 GPU 单层数据并绘制时间线。',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stage', choices=['decode'], default='decode', help='推理阶段')
    parser.add_argument('--model', choices=['kimi_k3'], default='kimi_k3', help='模型预设')
    parser.add_argument('--hardware', choices=['GB300Cluster8GPU'], default='GB300Cluster8GPU', help='每个逻辑节点固定使用 8-GPU GB300 NVLink 域')
    parser.add_argument('--nodes', type=int, default=1, help='同构集群节点数；每节点 8 GPU')
    parser.add_argument('--batch-size', type=int, default=8, help='全局 batch size')
    parser.add_argument('--context', type=int, default=8192, help='prefill 序列长度或 decode KV context 长度')
    parser.add_argument('--cp', type=int, default=1, help='Context Parallel')
    parser.add_argument('--dp', type=int, default=8, help='Data Parallel')
    parser.add_argument('--ep', type=int, default=8, help='Expert Parallel')
    parser.add_argument('--pp-partition', type=int, nargs='+', help='各 PP stage 的层数，省略则所有层放在同一 stage')
    parser.add_argument('--gpu-index', type=int, default=0, help='GPU 编号（0–7）')
    parser.add_argument('--layer', type=int, default=0, help='层编号（从 0 开始）')
    parser.add_argument('--output-dir', type=Path, default=OUT, help='结果目录，同名文件会覆盖')
    parser.add_argument('--print-config', action='store_true', help='仅检查并打印配置，不运行模拟')
    parser.add_argument('--two-lane', action='store_true', help='分析完成后绘制计算/HBM/通信时间线')
    parser.add_argument('--include-pp', action='store_true', help='包含本层输出侧 PP 边界事件（选 PP stage 最后一层）')
    args = parser.parse_args()
    model = load_model(args.model)
    args.pp_partition = args.pp_partition or [model.N_LAYERS]
    if any(getattr(args, k) <= 0 for k in ('batch_size', 'context', 'cp', 'dp', 'ep')):
        parser.error('batch/context/cp/dp/ep 必须为正整数')
    if any(n <= 0 for n in args.pp_partition) or sum(args.pp_partition) != model.N_LAYERS:
        parser.error(f'pp-partition 必须是正整数且总和为 {model.N_LAYERS}')
    n_gpus = 8 * args.nodes
    if args.nodes <= 0:
        parser.error('nodes 必须为正整数')
    if args.cp * args.dp != args.ep or args.ep * len(args.pp_partition) != n_gpus:
        parser.error(f'要求 cp * dp == ep 且 ep * PP stage 数 == GPU 数 {n_gpus}')
    if args.batch_size % (args.cp * args.dp) or model.N_EXPERTS % args.ep:
        parser.error('batch-size 必须能被 cp*dp 整除，专家数必须能被 ep 整除')
    if not 0 <= args.layer < model.N_LAYERS or not 0 <= args.gpu_index < n_gpus:
        parser.error(f'layer 范围为 0–{model.N_LAYERS-1}；gpu-index 范围为 0–{n_gpus-1}')
    window = getattr(model, 'WINDOW', None)
    if args.context % args.cp or window is not None and window % args.cp:
        parser.error('context 和 WINDOW（如模型定义）必须能被 cp 整除')
    for ratio in set(getattr(model, 'COMPRESS_RATIOS', ())) - {0}:
        if args.context % ratio or (args.context // ratio) % args.cp:
            parser.error(f'context 必须能被压缩倍率 {ratio} * cp 整除')
    layer_stages = [stage for stage, count in enumerate(args.pp_partition) for _ in range(count)]
    if args.gpu_index // args.ep != layer_stages[args.layer]:
        parser.error('选定 GPU 不属于该层的 PP stage，请调整 gpu-index 或 layer')
    return args, model


def main():
    global OUT, DEVICE, LAYER
    args, model = parse_args()
    is_prefill = args.stage == 'prefill'
    OUT = args.output_dir.resolve()
    LAYER = args.layer
    hw = GB300Cluster(nvl_scope=8, n_nodes=args.nodes)
    gpus = sorted([d for d in hw.nodes if isinstance(d, Compute) and d.kind == 'gpu'],
                  key=lambda d: (int(d.name.split('-', 1)[0][1:]),
                                 int(d.name.rsplit('-', 1)[1])))
    device = gpus[args.gpu_index]
    DEVICE = device.name
    memory = hw.find_local_memory(device)
    fabric = hw.find_fabric(device, memory)
    compress_ratios = getattr(model, 'COMPRESS_RATIOS', ())
    compression_ratio = (
        compress_ratios[LAYER] if LAYER < len(compress_ratios) else None)
    config = dict(model=args.model, hardware=args.hardware, stage=args.stage,
                  batch_size=args.batch_size, context=args.context, cp=args.cp, dp=args.dp,
                  ep=args.ep, pp_partition=args.pp_partition, gpu_index=args.gpu_index,
                  nodes=args.nodes, n_gpus=len(gpus),
                  device=DEVICE, layer=LAYER, model_layers=model.N_LAYERS,
                  compression_ratio=compression_ratio, measurement_start='read_input',
                  output_dir=str(OUT), data_source='fresh Simulator.run(), not an input JSON',
                  gpu_tflops=device.tflops, hbm_read_bandwidth_gbs=fabric.dst_to_src_bandwidth_gbs,
                  hbm_write_bandwidth_gbs=fabric.src_to_dst_bandwidth_gbs,
                  model_source=str(Path(model.__file__).resolve()),
                  include_pp=args.include_pp,
                  hardware_source=str(Path(sys.modules[type(hw).__module__].__file__).resolve()))
    print('实际模拟配置（其余模型/硬件常量来自下列预设源码）：', flush=True)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    if args.print_config:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    g, layers, emb, read, kv, head = model.declare_model(
        seq_prefill=args.context, decode=not is_prefill, batch_size=args.batch_size)
    if is_prefill:
        g, p = model.optimize_model_cluster_prefill(
            g, layers, hw, emb, read, head,
            cp=args.cp, dp=args.dp, ep=args.ep,
            pp_partition=args.pp_partition, n_gpus=len(gpus))
    else:
        g, p = model.optimize_model_cluster_decode(
            g, layers, hw, emb, read, kv, head, seq_prefill=args.context,
            cp=args.cp, dp=args.dp, ep=args.ep, pp_partition=args.pp_partition, n_gpus=len(gpus))
    layer = layers[LAYER]
    fields = {}
    if is_prefill:
        suffix = '_cp_dp_copies'
        for attr, copies in vars(layer).items():
            if attr.startswith('_') and attr.endswith(suffix):
                name = attr[1:-len(suffix)]
                for kernel in copies:
                    fields[kernel] = name
    else:
        for group in layer._decode_copies.values():
            for name, copies in group.items():
                for kernel in copies:
                    fields[kernel] = name
    for i, kernel in enumerate(layer.experts):
        fields[kernel] = f'expert_{i // 2}_{"up" if i % 2 == 0 else "down"}'
    if is_prefill:
        bridges = layer._bridge_cp_dp_copies
        ends = layer._ffn_add_cp_dp_copies
    else:
        def decode_copies(layer_meta, name):
            for group in ('dp', 'cp_dp'):
                if name in layer_meta._decode_copies[group]:
                    return layer_meta._decode_copies[group][name]
            raise KeyError(
                f'{name!r} not found in decode copies for layer {LAYER}')

        bridges = decode_copies(layer, 'bridge')
        ends = decode_copies(layer, 'ffn_add')
    # Intersection of descendants and ancestors also includes communications
    # introduced by graph transformations, without relying on event order.
    reachable = set(bridges)
    for k in bridges:
        reachable.update(nx.descendants(g._dag, k))
    preceding = set(ends)
    for k in ends:
        preceding.update(nx.ancestors(g._dag, k))
    layer_nodes = (reachable & preceding) | set(fields)
    # PP transfers are attached to the producers of the next stage's bridge.
    # With CP > 1 these can be Gather events combining CP gather + PP transfer.
    pp_nodes = set()
    boundaries = set()
    accumulated = 0
    for count in args.pp_partition[:-1]:
        accumulated += count
        boundaries.add(accumulated - 1)
    output_nodes = set()
    if LAYER + 1 < len(layers) and (LAYER not in boundaries or args.include_pp):
        next_bridges = (
            layers[LAYER + 1]._bridge_cp_dp_copies if is_prefill
            else decode_copies(layers[LAYER + 1], 'bridge'))
        for bridge in next_bridges:
            output_nodes.update(n for n in g._dag.predecessors(bridge)
                                if n not in ends and n in reachable)
        layer_nodes.update(output_nodes)
    if args.include_pp and LAYER in boundaries:
        next_bridges = (
            layers[LAYER + 1]._bridge_cp_dp_copies if is_prefill
            else decode_copies(layers[LAYER + 1], 'bridge'))
        for bridge in next_bridges:
            pp_nodes.update(g._dag.predecessors(bridge))
        layer_nodes.update(pp_nodes)
    elif args.include_pp:
        print(f'第 {LAYER} 层没有输出侧 PP 边界；可选择层 {sorted(boundaries)}。', flush=True)
    # IDs identify nodes in this optimized graph, including split copies and
    # inserted communication nodes; repeated operator types remain distinct.
    kernel_ids = {k: f'K{i:06d}' for i, k in enumerate(g.topological_sort())}
    result = Simulator(g, p, hw, measurement_start=read).run()
    local = [e for e in result.trace if e.device.name == DEVICE]
    local_bridges = [e for e in local if e.kernel in bridges]
    if local_bridges:
        begin = min(e.start_us for e in local_bridges)
    else:
        begin = min(
            e.start_us for e in local
            if e.kernel in layer_nodes
            and e.end_us > result.measurement_start_us)
    finish = max(e.end_us for e in local if e.kernel in ends)
    compute_finish = finish
    pp_entries = [e for e in local if e.kernel in pp_nodes]
    output_entries = [e for e in local if e.kernel in output_nodes | pp_nodes]
    if output_entries:
        finish = max(finish, max(e.end_us for e in output_entries))
    entries = sorted([e for e in local if e.kernel in layer_nodes
                      and e.start_us >= begin - 1e-7
                      and e.end_us <= finish + 1e-7],
                     key=lambda e: (e.start_us, e.end_us))
    window_entries = sorted(
        [e for e in local if e.end_us > e.start_us
         and e.start_us < finish and e.end_us > begin],
        key=lambda e: (e.start_us, e.end_us))
    # Kimi K3 carries auxiliary block/KDA state outside the ordinary
    # bridge-to-ffn graph path.  On the selected serial GPU, the actual
    # events inside the local bridge-to-ffn window are the authoritative
    # layer slice and include those auxiliary paths.
    entry_ids = {id(e) for e in entries}
    if {id(e) for e in window_entries} != {
            id(e) for e in entries if e.end_us > e.start_us}:
        entries = sorted(
            [*entries, *(e for e in window_entries if id(e) not in entry_ids)],
            key=lambda e: (e.start_us, e.end_us))
    # The additive breakdown below is valid for this serial, uncontended GPU
    # stream. Fail explicitly instead of silently using it for another setup.
    assert len({e.stream for e in entries}) == 1
    assert all(a.end_us <= b.start_us + 1e-7
               for a, b in zip(entries, entries[1:]))
    rows = []
    for e in entries:
        k = e.kernel
        field = fields.get(k, type(k).__name__)
        dur = e.end_us - e.start_us
        ct, mt = e.compute_time_us, e.memory_time_us
        assert abs(e.local_elapsed_time_us - max(ct, mt)) < 1e-6
        # This is a roofline attribution, not sequential physical phases.
        hbm_exposed = max(mt - ct, 0)
        net_exposed = max(dur - max(ct, mt), 0)
        assert abs(ct + hbm_exposed + net_exposed - dur) < 1e-6
        mem = hw.find_local_memory(e.device)
        fab = hw.find_fabric(e.device, mem)
        kv_bytes = 0.0
        weight_bytes = sum(float(t.size_bytes * k.weight_read_fraction)
                           for t in k.weights.values()
                           if (p.get_tensor_memory(t) or mem) is mem)
        if field == 'sa':
            for port, t in k.inputs.items():
                if 'kv' in port or 'index' in port:
                    if (p.get_tensor_memory(t) or mem) is mem:
                        kv_bytes += float(t.size_bytes * k.input_read_fraction(port))
        rows.append(dict(
            kernel_id=kernel_ids[k],
            predecessor_ids=' '.join(kernel_ids[n] for n in g._dag.predecessors(k)),
            device=DEVICE,
            field=field, operator=type(k).__name__,
            category=('PP boundary' if k in pp_nodes else
                      'Layer output gather' if k in output_nodes else category(field, k, e)),
            stream=e.stream,
            start_us=e.start_us - begin, end_us=e.end_us - begin,
            duration_us=dur, compute_us=ct, hbm_us=mt,
            network_base_us=e.network_time_us,
            network_elapsed_us=e.network_elapsed_time_us,
            hbm_exposed_us=hbm_exposed, network_exposed_us=net_exposed,
            attention_kv_read_bytes=kv_bytes,
            attention_kv_hbm_us=kv_bytes / (fab.dst_to_src_bandwidth_gbs * 1e3),
            weight_hbm_us=weight_bytes / (fab.dst_to_src_bandwidth_gbs * 1e3),
            weight_read_bytes=float(k.loaded_weight_bytes),
            bound=e.bound.value))
    with (OUT / 'events.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(lambda: defaultdict(float))
    metrics = ['duration_us', 'compute_us', 'hbm_us', 'network_base_us',
               'network_elapsed_us', 'hbm_exposed_us', 'network_exposed_us',
               'attention_kv_read_bytes', 'attention_kv_hbm_us', 'weight_read_bytes', 'weight_hbm_us']
    for row in rows:
        for metric in metrics:
            groups[row['category']][metric] += row[metric]
        groups[row['category']]['events'] += 1
    total = {m: sum(r[m] for r in rows) for m in metrics}
    idle = finish - begin - total['duration_us']
    assert idle >= -1e-6
    summary = dict(config=config, device=DEVICE, layer=LAYER,
                   compression_ratio=compression_ratio,
                   layer_span_us=finish - begin, idle_us=max(idle, 0),
                   layer_compute_window_us=compute_finish - begin,
                   layer_output_event_us=sum(e.end_us - e.start_us for e in output_entries),
                   pp_boundary_event_us=sum(e.end_us - e.start_us for e in pp_entries),
                   pp_boundary_note='Whole boundary event; may combine CP gather and PP transfer, not isolated PP link time',
                   start_after_measurement_us=begin - result.measurement_start_us,
                   full_run_us=result.measured_time_us,
                   pre_measurement_us=result.measurement_start_us,
                   full_model_kv_preload_us=(result.measurement_start_us if not is_prefill else 0.0),
                   zero_duration_event_count=sum(r['duration_us'] == 0 for r in rows),
                   event_count=len(rows), categories=dict(groups), totals=total)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    (OUT / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
    write_report(rows, summary)
    plot(rows, summary)
    print(f'{args.stage} ({args.hardware}): {result.measured_time_us:.3f} us', flush=True)
    print(f'GPU {args.gpu_index}, layer {LAYER}: {finish - begin:.3f} us', flush=True)
    if args.two_lane:
        subprocess.run([sys.executable, str(Path(__file__).with_name('plot_two_lane_timeline.py')),
                        '--input-dir', str(OUT)], check=True)
    print(f'输出目录：{OUT}', flush=True)


def plot(rows, s):
    colors = ['#5b7cfa', '#21a7a0', '#8c66d9', '#e3a125', '#dc738a', '#ef6c45', '#64748b']
    cats = list(s['categories'])
    palette = {cat: colors[i % len(colors)] for i, cat in enumerate(cats)}
    c = s["config"]
    fig, axes = plt.subplots(4, 1, figsize=(14, 15),
                             gridspec_kw={'height_ratios': [1.8, 2, 1.2, 1.2]})
    fig.suptitle(f'{c["model"]} {c["stage"]} | GPU {c["gpu_index"]} | Layer {s["layer"]}',
                 fontsize=19, fontweight='bold', x=.08, ha='left')
    fig.text(.08, .94, f'{c["hardware"]} | batch {c["batch_size"]} | CP {c["cp"]} / DP {c["dp"]} / EP {c["ep"]} | context {c["context"]} | '
             f'layer span {s["layer_span_us"]:.3f} us | simulator estimates', fontsize=11)
    ax = axes[0]
    for r in rows:
        y = cats.index(r['category'])
        ax.broken_barh([(r['start_us'], r['duration_us'])], (y-.32, .64),
                      facecolors=palette[r['category']])
    ax.set_yticks(range(len(cats)), cats)
    ax.invert_yaxis()
    ax.set_xlim(0, s['layer_span_us'])
    ax.set_xlabel('Time since layer input (us); one serial GPU stream')
    ax.set_title('A. Operator timeline', loc='left', fontweight='bold')
    ax.grid(axis='x', alpha=.2)
    ax = axes[1]
    for i, cat in enumerate(cats):
        v = s['categories'][cat]
        left = 0
        for key, label, color in [('compute_us', 'Compute demand', '#5b7cfa'),
                                  ('hbm_exposed_us', 'HBM beyond compute', '#e3a125'),
                                  ('network_exposed_us', 'Network beyond local work', '#ef6c45')]:
            ax.barh(i, v[key], left=left, color=color, label=label if i == 0 else None)
            left += v[key]
        ax.text(left + .12, i, f'{left:.3f} us', va='center', fontsize=10)
    ax.set_yticks(range(len(cats)), cats)
    ax.invert_yaxis()
    ax.set_xlim(0, max(v['duration_us'] for v in s['categories'].values()) * 1.25)
    ax.set_xlabel('Additive roofline attribution (us), not sequential hardware phases')
    ax.set_title('B. Operator elapsed time and exposed resource cost', loc='left', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=.2)
    ax = axes[2]
    t = s['totals']
    labels = ['Compute demand', 'HBM beyond compute', 'Network beyond local work', 'Idle / wait']
    values = [t['compute_us'], t['hbm_exposed_us'], t['network_exposed_us'], s['idle_us']]
    left = 0
    for label, value, color in zip(labels, values, ['#5b7cfa', '#e3a125', '#ef6c45', '#d1d5db']):
        ax.barh(0, value, left=left, color=color, label=f'{label}: {value:.3f} us')
        if value > s['layer_span_us'] * .08:
            ax.text(left + value/2, 0, f'{value:.2f} us', ha='center', va='center', color='white', fontweight='bold')
        left += value
    ax.set_xlim(0, s['layer_span_us'])
    ax.set_yticks([])
    ax.set_xlabel('Layer span (us)')
    ax.set_title('C. Whole-layer attribution', loc='left', fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.35), ncol=2, fontsize=9)
    ax = axes[3]
    values = [t['weight_hbm_us'], t['attention_kv_hbm_us'],
              t['hbm_us'] - t['weight_hbm_us'] - t['attention_kv_hbm_us']]
    left = 0
    for label, value, color in zip(['Weights', 'Attention KV read', 'Other HBM traffic'],
                                   values, ['#e3a125', '#21a7a0', '#64748b']):
        ax.barh(0, value, left=left, color=color, label=f'{label}: {value:.4f} us')
        left += value
    ax.set_xlim(0, t['hbm_us'])
    ax.set_yticks([])
    ax.set_xlabel('HBM service demand (us); overlaps compute, not added to layer span')
    ax.set_title('D. HBM demand: weights vs KV vs other traffic', loc='left', fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.35), ncol=3, fontsize=9)
    stage_note = (f'Full-model KV preload ({s["full_model_kv_preload_us"]:.1f} us) is outside this decode-layer window. '
                  if c['stage'] == 'decode' else
                  f'Prefill sequence length: {c["context"]}; pre-measurement time: {s["pre_measurement_us"]:.1f} us. ')
    fig.text(.08, .025,
             f'Attention KV read: {t["attention_kv_read_bytes"]/1024:.1f} KiB / {t["attention_kv_hbm_us"]:.4f} us HBM demand (included above).\n'
             'HBM and compute overlap within each kernel. Exposed HBM = max(HBM - compute, 0).\n'
             'Network exposure = elapsed - max(compute, HBM); not a global counterfactual speedup.\n'
             + stage_note + f'Compression ratio: {s["compression_ratio"]}.',
             fontsize=10, linespacing=1.5)
    fig.subplots_adjust(left=.22, right=.95, top=.90, bottom=.17, hspace=.95)
    fig.savefig(OUT / f'gpu{c["gpu_index"]}_layer{s["layer"]}.png', dpi=180)
    fig.savefig(OUT / f'gpu{c["gpu_index"]}_layer{s["layer"]}.svg')
    plt.close(fig)


def write_report(rows, s):
    c = s['config']
    stem = f'gpu{c["gpu_index"]}_layer{s["layer"]}'
    lines = [f'# GPU {c["gpu_index"]} / Layer {s["layer"]}', '',
             '数据来自本次模拟，不是 GPU 实测；完整参数见 config.json。', '',
             '```json', json.dumps(c, ensure_ascii=False, indent=2), '```', '',
             f'层时间窗口：{s["layer_span_us"]:.6f} μs；完整 {c["stage"]}：{s["full_run_us"]:.6f} μs。',
             (f'KV preload：{s["full_model_kv_preload_us"]:.6f} μs（不计入 decode）。'
              if c['stage'] == 'decode' else
              f'测量起点前耗时：{s["pre_measurement_us"]:.6f} μs。'), '',
             f'![耗时拆分]({stem}.png)', '',
             '| 类别 | 事件数 | 算子耗时 μs | 计算需求 μs | HBM 需求 μs | HBM 暴露 μs | 网络额外暴露 μs |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for cat, v in s['categories'].items():
        lines.append(f'| {cat} | {int(v["events"])} | {v["duration_us"]:.6f} | {v["compute_us"]:.6f} | {v["hbm_us"]:.6f} | {v["hbm_exposed_us"]:.6f} | {v["network_exposed_us"]:.6f} |')
    lines += ['', '已验证本次 GPU 事件串行且本地 elapsed 等于 max(计算需求, HBM 需求)。',
              'HBM 暴露 = max(HBM需求 − 计算需求, 0)；网络额外暴露 = elapsed − max(计算需求, HBM需求)。',
              '上述资源分解不是硬件顺序阶段。双轨图使用实际算子起止时间，计算轨包含 HBM 开销。',
              'KV 输入 HBM 时间已包含于算子 HBM 需求，不应再次相加。',
              f'Attention KV 读取：{s["totals"]["attention_kv_read_bytes"]:.0f} bytes；HBM需求 {s["totals"]["attention_kv_hbm_us"]:.6f} μs。',
              '详细事件见 events.csv，汇总见 summary.json。kernel_id 对应本次优化后计算图的拓扑编号，predecessor_ids 记录前驱节点。',
              f'[逐 kernel 分页时间线]({stem}_kernels.pdf) · [完整明细 SVG]({stem}_kernels.svg)',
              f'明细包含选定 GPU、本层 {c["stage"]} 窗口中的零耗时节点；其他 GPU 的 kernel 和测量起点前耗时不在此窗口。', '']
    (OUT / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
