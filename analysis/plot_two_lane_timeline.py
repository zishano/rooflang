"""Plot recorded compute, HBM, and communication intervals on one time axis."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from kernel_timeline_detail import export_kernel_detail
from timing_metrics import overlap_duration
from communication_metric_plots import export_metric_plots


parser = argparse.ArgumentParser(description='从事件数据目录绘制计算/HBM/通信时间线')
parser.add_argument('--input-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'results/layer_timing')
parser.add_argument('--details', action='store_true', help='额外生成逐 kernel 明细图和分页 PDF')
args = parser.parse_args()
OUT = args.input_dir.resolve()
summary = json.loads((OUT / 'summary.json').read_text())
config = summary.get('config', {})
gpu_index = config.get('gpu_index', int(summary['device'].rsplit('-', 1)[1]))
layer_index = summary['layer']
with (OUT / 'events.csv').open() as f:
    rows = list(csv.DictReader(f))
for r in rows:
    for key in ('start_us', 'end_us', 'duration_us', 'network_base_us', 'compute_us', 'hbm_us'):
        r[key] = float(r[key])
comm = [r for r in rows if r['network_base_us'] > 0]
compute = [r for r in rows if r['network_base_us'] == 0]
compute_intervals = [(r['start_us'], r['end_us']) for r in compute]
overlap = overlap_duration(compute_intervals, [(r['start_us'], r['end_us']) for r in comm])
communication_metrics = []
for r in comm:
    r['overlapped_us'] = overlap_duration(compute_intervals, [(r['start_us'], r['end_us'])])
    communication_metrics.append(dict(
        kernel_id=r.get('kernel_id', ''), field=r['field'], operator=r['operator'],
        category=r['category'], start_us=r['start_us'], end_us=r['end_us'],
        original_us=r['network_base_us'], elapsed_us=r['duration_us'],
        overlapped_us=r['overlapped_us']))
with (OUT / 'communication_timing.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=['kernel_id', 'field', 'operator', 'category',
        'start_us', 'end_us', 'original_us', 'elapsed_us', 'overlapped_us'])
    writer.writeheader()
    writer.writerows(communication_metrics)
with (OUT / 'hbm_timing.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=['kernel_id', 'field', 'operator', 'category',
        'start_us', 'compute_us', 'hbm_us', 'elapsed_us'])
    writer.writeheader()
    writer.writerows(dict(kernel_id=r.get('kernel_id', ''), field=r['field'],
        operator=r['operator'], category=r['category'], start_us=r['start_us'],
        compute_us=r['compute_us'], hbm_us=r['hbm_us'], elapsed_us=r['duration_us'])
        for r in rows if r['hbm_us'] > 0)
span = summary['layer_span_us']
offset = summary.get('start_after_measurement_us', summary.get('start_after_decode_us', 0.0))
palette = {
    'Attention projections': '#2563eb', 'Attention core': '#06b6d4',
    'MoE gate': '#b48b20', 'Shared expert': '#8b5cf6',
    'MoE experts': '#0d9488', 'Small / local ops': '#64748b',
    'Communication': '#ef4444',
    'PP boundary': '#c026d3',
    'Layer output gather': '#ea580c',
}
def display_group(row):
    if row['category'] in ('PP boundary', 'Layer output gather'):
        return row['category']
    if row['network_base_us'] > 0:
        return 'Communication'
    return row['category'] if row['category'] in palette else 'Small / local ops'

fig, ax = plt.subplots(figsize=(18, 8.5))
fig.patch.set_facecolor('#ffffff')
stage_name = config.get('stage', 'decode')
full_run_us = summary.get('full_run_us', summary.get('full_decode_us', 0.0))
fig.suptitle(f'GPU {gpu_index} | Layer {layer_index} {stage_name}: compute and communication on one timeline',
             x=.10, y=.98, ha='left', fontsize=18, fontweight='bold')
fig.text(.10, .90,
         f'{"Layer + PP boundary" if summary.get("pp_boundary_event_us", 0) else "Layer + output gather" if summary.get("layer_output_event_us", 0) else "Layer duration"}: {span:.3f} us    |    Compute-op / communication-op overlap: {overlap:.3f} us    |    Simulator trace',
         fontsize=12)
for y in (1, 0, -1.8):
    ax.add_patch(Rectangle((0, y-.18), span, .36, facecolor='#f1f5f9', edgecolor='#cbd5e1', lw=.7))
for event_index, r in enumerate(rows):
    r.setdefault('kernel_id', f'event_{event_index:04d}')
    is_comm = r['network_base_us'] > 0
    y = -1.8 if is_comm else 1
    color = palette[display_group(r)]
    if is_comm:
        rect = Rectangle((r['start_us'], y-.18), r['duration_us'], .36,
                         facecolor=color, edgecolor='none', alpha=.35)
        rect.set_gid('kernel-' + r['kernel_id'])
        ax.add_patch(rect)
    elif r['compute_us'] > 0:
        compute_bar = Rectangle((r['start_us'], y-.18), r['compute_us'], .36,
                                facecolor=color, edgecolor='#0f172a', linewidth=.35,
                                alpha=1.0, zorder=5)
        compute_bar.set_gid('kernel-' + r['kernel_id'])
        ax.add_patch(compute_bar)
        # Keep very short compute demands findable without stretching their duration.
        ax.vlines(r['start_us'] + r['compute_us'], y-.18, y+.18,
                  color='#0f172a', linewidth=1.0, zorder=6)
    if (is_comm and r['duration_us'] == 0) or (not is_comm and r['compute_us'] == 0):
        ax.plot(r['start_us'], y, '|', color='#64748b', markersize=6)
# HBM service demand is shown on its own middle lane. It starts with the
# operator event and can overlap the compute lane; it is not an extra phase
# added after compute. The simulator's local elapsed time is max(compute, HBM).
for r in rows:
    if r['hbm_us'] <= 0:
        continue
    mem_duration = min(r['hbm_us'], r['duration_us'])
    ax.add_patch(Rectangle((r['start_us'], -.18), mem_duration, .36,
                           facecolor='#e3a125', edgecolor='none', alpha=.9,
                           gid='hbm-' + r['kernel_id']))
    if mem_duration == 0:
        ax.plot(r['start_us'], 0, '|', color='#a16207', markersize=6)
# Label each kernel field at its actual start time. Repeated expert kernels are
# grouped by operation, and labels with nearby starts are stacked onto free rows.
def kernel_labels(source_rows, demand_key, include_communication=False):
    grouped = {}
    for row in source_rows:
        if row[demand_key] <= 0 or (row['network_base_us'] > 0 and not include_communication):
            continue
        key = (f'expert_*_{row["field"].rsplit("_", 1)[-1]}', row['operator']) if row['field'].startswith('expert_') else (row['field'], row['operator'])
        item = grouped.setdefault(key, {'row': row, 'value': 0.0, 'count': 0, 'field': key[0]})
        item['value'] = max(item['value'], row[demand_key])
        item['count'] += 1
        if row['start_us'] < item['row']['start_us']:
            item['row'] = row
    items = list(grouped.values())
    cutoff = max((item['value'] for item in items), default=0) * .2
    for item in items:
        item['show_time'] = item['value'] >= cutoff
        item['name'] = item['field']
        if item['field'].startswith('expert_*_'):
            item['name'] += f' ×{item["count"]}'
        if item['show_time']:
            item['name'] += f' ({item["value"]:.3g} us)'
        item['x'] = item['row']['start_us']
    return sorted(items, key=lambda item: item['x'])

def place_start_labels(items, lane_y, side, color, fontsize=6.8):
    occupied = []
    for item in items:
        x = item['x']
        # Approximate label width in data units; keep the anchor exactly at start.
        half_width = max(span * .012, span * len(item['name']) * .0018)
        interval = (max(0, x-half_width), min(span, x+half_width))
        row = 0
        while any(row == used_row and interval[0] < used[1] and used[0] < interval[1]
                  for used, used_row in occupied):
            row += 1
        occupied.append((interval, row))
        if side == 'above':
            label_y = lane_y + .38 + row*.20
            ax.vlines(x, lane_y, label_y-.02, color=color, linewidth=.6, zorder=4)
            ax.text(x, label_y, item['name'], ha='center', va='bottom',
                    fontsize=fontsize, color=color)
        else:
            label_y = lane_y - .30 - row*.17
            ax.vlines(x, label_y+.03, lane_y, color=color, linewidth=.6, zorder=4)
            ax.text(x, label_y, item['name'], ha='center', va='top',
                    fontsize=fontsize, color=color)

compute_labels = kernel_labels(compute, 'compute_us')
hbm_labels = kernel_labels(rows, 'hbm_us', include_communication=True)
place_start_labels(compute_labels, 1.18, 'above', '#1d4ed8')
place_start_labels(hbm_labels, -.18, 'below', '#92400e', fontsize=6.5)
for i, r in enumerate(comm):
    mid = (r['start_us']+r['end_us'])/2
    comm_color = palette[display_group(r)]
    event_name = ('PP boundary / ' if r['category'] == 'PP boundary' else
                  'Layer output / ' if r['category'] == 'Layer output gather' else '') + r['operator']
    ax.plot([mid, mid], [-1.60, .80], color=comm_color, lw=.6, alpha=.6, linestyle=':')
    label = (f'{event_name}\nO: {r["network_base_us"]:.4f} | E: {r["duration_us"]:.4f} us'
             f'\nOverlap: {r["overlapped_us"]:.4f} us')
    if len(comm) > 2:
        text_x = span * (i + .5) / len(comm)
        align = 'center'
    else:
        text_x = mid - 1 if i == 0 else span - .3
        align = 'center' if i == 0 else 'right'
    ax.annotate(label, xy=(mid, -1.60), xytext=(text_x, -1.85),
                ha=align, va='top', fontsize=9,
                color=comm_color, arrowprops={'arrowstyle':'-', 'color':comm_color})
ax.set_xlim(0, span)
ax.set_ylim(-3.0, 3.0)
ax.set_yticks([1, 0, -1.8], ['Compute demand', 'HBM / memory demand', 'Communication ops'])
ax.tick_params(axis='y', length=0, labelsize=12, pad=12)
ax.set_xlabel('Time since layer input (us)', fontsize=12, labelpad=8)
ax.set_xticks([span*i/5 for i in range(6)], [f'{span*i/5:.3f}' for i in range(6)])
ax.grid(axis='x', color='#e2e8f0', alpha=.7)
ax.set_axisbelow(True)
for spine in ['left', 'top', 'right']:
    ax.spines[spine].set_visible(False)
# The legend is derived from this run, including all tiny/zero-time kernels.
# Group colors do not move kernels: their original start/end times are kept.
handles = []
for group, color in palette.items():
    members = [r for r in rows if display_group(r) == group]
    if not members:
        continue
    elapsed = sum(r['compute_us'] for r in members if r['network_base_us'] == 0)
    field_names = list(dict.fromkeys(r['field'] for r in members))
    if group == 'MoE experts':
        member_names = 'expert_*_up/down'
    elif len(field_names) > 5:
        member_names = ', '.join(field_names[:5]) + ', …'
    else:
        member_names = ', '.join(field_names)
    label = f'{group}: {member_names} (compute {elapsed:.3f} us)'
    handles.append(Patch(color=color, label=label))
handles.append(Patch(facecolor='white', edgecolor='#0f172a', label='Upper bar = compute_us'))
handles.append(Patch(color='#e3a125', label='HBM demand (hbm_us)'))
fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.15, .22, .82, .20),
           ncol=3, mode='expand', frameon=False, fontsize=8.5, labelspacing=.65,
           columnspacing=1.2, handlelength=1.5, borderaxespad=0)
fig.text(.15, .135,
         'Compute and HBM bars show separate demands from each kernel start; they overlap, not sequential phases. Communication bars show actual event duration. O = base network estimate; E = event duration; overlap = time shared with local ops.',
         fontsize=8.5)
fig.text(.15, .095,
         f'Layer window [{offset:.3f}, {offset+span:.3f}] us of {full_run_us:.3f} us {stage_name}. HBM kernel details: hbm_timing.csv and SVG bar tooltips.',
         fontsize=8.5)
fig.subplots_adjust(left=.15, right=.98, bottom=.39, top=.85)
fig.savefig(OUT / f'gpu{gpu_index}_layer{layer_index}_two_lane.png', dpi=200)
fig.savefig(OUT / f'gpu{gpu_index}_layer{layer_index}_two_lane.svg')
print(f'Layer span {span:.6f} us; cross-lane overlap {overlap:.6f} us')
for r in comm:
    print(r['operator'], r['start_us'], r['end_us'])

# Native SVG tooltips preserve each event identity even at overview scale.
import xml.etree.ElementTree as ET
svg_path = OUT / f'gpu{gpu_index}_layer{layer_index}_two_lane.svg'
root = ET.parse(svg_path)
ns = 'http://www.w3.org/2000/svg'
by_id = {'kernel-' + r['kernel_id']: r for r in rows}
for element in root.iter():
    row = by_id.get(element.get('id'))
    if row is None and element.get('id', '').startswith(('hbm-', 'compute-')):
        row = by_id.get('kernel-' + element.get('id').split('-', 1)[1])
    if row is not None:
        title = ET.Element('{' + ns + '}title')
        title.text = (f'{row["kernel_id"]}: {row["field"]} / {row["operator"]}; '
                      f'start={row["start_us"]:.6f} us, end={row["end_us"]:.6f} us, '
                      f'duration={row["duration_us"]:.6f} us, compute={row["compute_us"]:.6f} us, '
                      f'hbm={row["hbm_us"]:.6f} us, stream={row["stream"]}')
        element.insert(0, title)
root.write(svg_path, encoding='utf-8', xml_declaration=True)
export_metric_plots(rows, summary, OUT, palette, display_group)
if args.details:
    export_kernel_detail(rows, summary, OUT)
