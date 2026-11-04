"""Three aligned views of communication original/elapsed/overlapped time."""
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from timing_metrics import intersection_intervals


def export_metric_plots(rows, summary, out, palette, display_group):
    config = summary.get('config', {})
    gpu = config.get('gpu_index', int(summary['device'].rsplit('-', 1)[1]))
    layer = summary['layer']
    span = summary['layer_span_us']
    compute = [r for r in rows if r['network_base_us'] == 0]
    comm = [r for r in rows if r['network_base_us'] > 0]
    compute_intervals = [(r['start_us'], r['end_us']) for r in compute]
    definitions = {
        'original': ('Original network time',
                     'Isolated network estimate; bars are anchored at actual event starts for comparison, not a resimulated schedule.'),
        'elapsed': ('Elapsed communication-event time',
                    'Actual simulated intervals: end - start. Includes contention during execution; excludes waiting before event start.'),
        'overlapped': ('Communication overlapped with other compute/local ops',
                       'Intersection of actual communication and other compute/local intervals on this GPU; not intra-kernel overlap.'),
    }
    # Identical bounds in all three figures, including isolated estimates if
    # they ever extend beyond the observed layer window.
    axis_end = max([span] + [r['start_us'] + r['network_base_us'] for r in comm])
    for metric, (title, explanation) in definitions.items():
        fig, ax = plt.subplots(figsize=(18, 8.5))
        fig.suptitle(f'GPU {gpu} | Layer {layer} | {title}', x=.10, y=.98,
                     ha='left', fontsize=18, fontweight='bold')
        fig.text(.10, .92, explanation, fontsize=11)
        fig.text(.10, .875, 'Upper lane is the same actual compute/local timeline in all three views (including HBM time).', fontsize=10)
        for y in (0, 1):
            ax.add_patch(Rectangle((0, y-.18), axis_end, .36, color='#f1f5f9'))
        for r in compute:
            color = palette[display_group(r)]
            ax.add_patch(Rectangle((r['start_us'], .82), r['duration_us'], .36,
                                   color=color, lw=0))
            if r['duration_us'] == 0:
                ax.plot(r['start_us'], 1, '|', color='#64748b', markersize=5)
            if r['field'] in ('wq_b', 'wo_a', 'wo_b', 'sw_up', 'sw_down'):
                ax.text((r['start_us']+r['end_us'])/2, 1, r['field'],
                        va='center', ha='center', color='white', fontsize=10)
        experts = [r for r in compute if r['category'] == 'MoE experts']
        if experts:
            ax.text((min(r['start_us'] for r in experts)+max(r['end_us'] for r in experts))/2,
                    1, 'MoE experts', color='white', va='center', ha='center')
        total = 0.0
        for i, r in enumerate(comm):
            color = palette[display_group(r)]
            if metric == 'original':
                intervals = [(r['start_us'], r['start_us']+r['network_base_us'])]
            elif metric == 'elapsed':
                intervals = [(r['start_us'], r['end_us'])]
            else:
                intervals = intersection_intervals([(r['start_us'], r['end_us'])], compute_intervals)
            duration = sum(end-start for start, end in intervals)
            total += duration
            for start, end in intervals:
                ax.add_patch(Rectangle((start, -.18), end-start, .36, color=color, lw=0))
            name = ('PP / ' if r['category'] == 'PP boundary' else
                    'Output / ' if r['category'] == 'Layer output gather' else '') + r['operator']
            label = f'{name}\n{duration:.6f} us'
            text_x = axis_end*(i+.5)/len(comm)
            if duration:
                center = (intervals[0][0]+intervals[-1][1])/2
                ax.annotate(label, xy=(center, -.18), xytext=(text_x, -.45),
                            ha='center', va='top', color=color, fontsize=9,
                            arrowprops={'arrowstyle':'-', 'color':color})
            else:
                ax.text(text_x, -.45, label, ha='center', va='top', color=color, fontsize=9)
        if metric == 'overlapped' and total < 1e-9:
            ax.text(axis_end/2, 0, 'No overlap: 0.000000 us', ha='center', va='center', color='#334155', fontsize=13)
        ax.set(xlim=(0, axis_end), ylim=(-1.05, 1.55), yticks=[0,1],
               yticklabels=['Communication', 'Compute / local'], xlabel='Time since layer input (us)')
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
        for side in ('left','right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis='y', length=0, pad=10)
        ax.set_title(f'{metric.capitalize()} per-event sum: {total:.6f} us | Layer window: {span:.6f} us',
                     loc='left', fontsize=11, pad=14)
        handles=[]
        for group, color in palette.items():
            members=[r for r in rows if display_group(r)==group]
            if not members:
                continue
            types=', '.join(sorted({r['operator'] for r in members}))
            if group=='Small / local ops':
                types += '; includes moe_add / ffn_add, local copies'
            if group=='PP boundary':
                types += '; whole boundary event, not isolated PP cost'
            handles.append(Patch(color=color,label=group+'\n'+textwrap.fill(types,78)))
        fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.10,.37),
                   ncol=2, frameon=False, fontsize=10, labelspacing=1.1, columnspacing=3)
        fig.text(.10,.045,'Original is network_base_us; elapsed is event duration. They are not related by original - overlap = elapsed.', fontsize=10)
        fig.subplots_adjust(left=.10,right=.97,top=.80,bottom=.47)
        for extension in ('png','svg'):
            fig.savefig(out/f'gpu{gpu}_layer{layer}_communication_{metric}.{extension}',dpi=180)
        plt.close(fig)
