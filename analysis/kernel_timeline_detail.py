"""Per-kernel Gantt detail using the same physical time axis as the two lanes."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle


def export_kernel_detail(rows, summary, out):
    """One row per trace event, including zero-duration structural nodes.

    Export a full PNG/SVG and a paginated PDF. No aggregation by operator type.
    IDs are topological node IDs of this simulation's optimized graph.
    """
    out = Path(out)
    config = summary.get('config', {})
    gpu = config.get('gpu_index', int(summary['device'].rsplit('-', 1)[1]))
    layer = summary['layer']
    stem = f'gpu{gpu}_layer{layer}_kernels'
    span = summary['layer_span_us']
    for i, r in enumerate(rows):
        r.setdefault('kernel_id', f'event_{i:04d}')

    def draw(selected, first=0, page=None):
        n = len(selected)
        fig = plt.figure(figsize=(22, 3.6 + n * .25))
        grid = fig.add_gridspec(2, 3, height_ratios=[2, max(2, n)],
                               width_ratios=[5.5, 8, 5], hspace=.24, wspace=.04)
        overview = fig.add_subplot(grid[0, :])
        names = fig.add_subplot(grid[1, 0])
        gantt = fig.add_subplot(grid[1, 1])
        table = fig.add_subplot(grid[1, 2])
        title = f'GPU {gpu} | Layer {layer} | Every kernel | {len(rows)} events'
        if page is not None:
            title += f' | Page {page}'
        fig.suptitle(title, x=.03, ha='left', fontsize=16, fontweight='bold', y=.995)
        fig.text(.03, .995 - .25 / fig.get_figheight(),
                 f'{config.get("model", "")} / {config.get("hardware", "")} | '
                 f'Layer {span:.6f} us | Times relative to layer input | Simulator estimates', fontsize=10, va='top')
        for y in [0, 1]:
            overview.add_patch(Rectangle((0, y-.28), span, .56, color='#f1f5f9'))
        for r in rows:
            is_comm = r['network_base_us'] > 0
            y = 0 if is_comm else 1
            color = '#dc2626' if is_comm else '#2563eb'
            if r['duration_us'] == 0:
                overview.plot(r['start_us'], y, '|', color='#6b7280', markersize=5)
            else:
                overview.add_patch(Rectangle((r['start_us'], y-.28), r['duration_us'], .56, color=color, lw=0))
        overview.set(xlim=(0, span), ylim=(-.6, 1.6), yticks=[0, 1],
                     yticklabels=['Communication', 'Compute / local'], xlabel='Time (us)')
        overview.grid(axis='x', alpha=.2)
        for ax in (names, gantt, table):
            ax.set_ylim(n-.5, -.5)
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            for i in range(n):
                if i % 2 == 0:
                    ax.axhspan(i-.5, i+.5, color='#f1f5f9', zorder=0)
        names.set_xlim(0, 1)
        table.set_xlim(0, 1)
        names.set_xticks([])
        table.set_xticks([])
        gantt.set_xlim(0, span)
        gantt.xaxis.tick_top()
        gantt.grid(axis='x', alpha=.2)
        names.set_title('No. / Graph node ID / Field / Kernel type', loc='left', fontsize=10, pad=20)
        gantt.set_title('Actual start / end on common time axis (us)', fontsize=10, pad=20)
        table.set_title(f'{"Str":>3}     {"Start (us)":>13}  {"End (us)":>13}  {"Duration (us)":>13}',
                        loc='left', fontsize=8, pad=20, fontfamily='monospace')
        for i, r in enumerate(selected):
            is_comm = r['network_base_us'] > 0
            color = '#dc2626' if is_comm else '#2563eb'
            label = f'{first+i+1:03d}  {r["kernel_id"]}  {r["field"]} / {r["operator"]}'
            names.text(.005, i, label, fontsize=8, va='center', color='#111827')
            if r['duration_us'] == 0:
                gantt.plot(r['start_us'], i, marker='D', color='#64748b', markersize=3)
            else:
                gantt.barh(i, r['duration_us'], left=r['start_us'], height=.62, color=color)
                # A start tick makes subpixel events visible without stretching
                # the duration bar or distorting the physical time axis.
                gantt.plot(r['start_us'], i, '|', color=color, markersize=5)
            values = f'{int(r["stream"]):>3}     {r["start_us"]:>13.6f}  {r["end_us"]:>13.6f}  {r["duration_us"]:>13.6f}'
            table.text(.005, i, values, fontsize=8, va='center', fontfamily='monospace')
        fig.text(.03, .008, 'Blue: compute/local ops (includes HBM). Red: network ops. Diamond: zero-duration kernel. '
                 'Each row is one event; no expert/kernel aggregation. Tiny kernels retain true duration.', fontsize=9)
        fig.subplots_adjust(left=.08, right=.99, top=1-.85/fig.get_figheight(), bottom=.5/fig.get_figheight())
        return fig

    fig = draw(rows)
    fig.savefig(out / f'{stem}.png', dpi=130)
    fig.savefig(out / f'{stem}.svg')
    plt.close(fig)
    with PdfPages(out / f'{stem}.pdf') as pdf:
        for start in range(0, len(rows), 36):
            fig = draw(rows[start:start+36], start, start//36+1)
            pdf.savefig(fig)
            if start == 0:
                fig.savefig(out / f'{stem}_page1.png', dpi=130)
            plt.close(fig)
    print(f'逐 kernel 时间线：{out / (stem + ".pdf")} ({len(rows)} events, including '
          f'{sum(r["duration_us"] == 0 for r in rows)} zero-duration kernels)')
