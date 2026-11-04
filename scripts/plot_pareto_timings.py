#!/usr/bin/env python3
"""Plot three timing frontiers, their union frontier, and dominated candidates.

Read existing raw_results.jsonl files only; never run simulations or modify inputs.
Write static PNG/PDF figures and a self-contained interactive HTML figure.
Hover over original or overlapped frontier points to inspect their parameters.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from rooflang.programs.experiments.find_pareto_frontier import (  # noqa: E402
    THROUGHPUT_METRICS, MEMORY_FEASIBILITY_FIELDS, pareto_frontier,
)


def prepare(records):
    """Return feasible projections, per-timing frontiers, and their union frontier."""
    candidates, frontiers = {}, {}
    for timing, (xfield, yfield) in THROUGHPUT_METRICS.items():
        memory = MEMORY_FEASIBILITY_FIELDS[timing]
        points = []
        for record in records:
            if record.get('status') != 'ok' or (memory and not record.get(memory, False)):
                continue
            x, y = record.get(xfield), record.get(yfield)
            if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (x, y)):
                continue
            points.append(dict(record, x=x, y=y, timing=timing))
        candidates[timing] = points
        frontiers[timing] = pareto_frontier(points, 'x', 'y')
    frontiers['combined'] = pareto_frontier(
        [p for points in candidates.values() for p in points], 'x', 'y')
    return candidates, frontiers


def read_groups(source):
    if source.is_file():
        paths = [source]
    else:
        paths = ([source / 'raw_results.jsonl'] if (source / 'raw_results.jsonl').is_file()
                 else sorted(source.glob('*/raw_results.jsonl')))
    if not paths:
        raise ValueError(f'No raw_results.jsonl found in {source}')
    groups = {}
    for path in paths:
        latest = {}
        # Read a fixed prefix if a simulation is still appending to the file.
        data = path.read_bytes()
        lines = data.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if i == len(lines) - 1 and not line.endswith(b'\n'):
                    print(f'Warning: ignoring incomplete final line in {path}', file=sys.stderr)
                    break
                raise ValueError(f'Invalid JSON at {path}:{i+1}')
            latest[record['case_id']] = record
        for record in latest.values():
            key = (record.get('model') or path.parent.name, record['workload'],
                   record['hardware'], record['n_gpus'])
            groups.setdefault(key, []).append(record)
    return groups, paths


def write_interactive_html(key, candidates, frontiers, out, stem, scale):
    """Write a dependency-free interactive SVG plot with parameter tooltips."""
    model, workload, hardware, gpus = key
    grey = []
    for timing, points in candidates.items():
        kept = {p['case_id'] for p in frontiers[timing]}
        grey.extend(p for p in points if p['case_id'] not in kept)
    payload = {
        'title': f'{model} | {workload} | {hardware.upper()} | {gpus} GPUs',
        'scale': scale,
        'candidates': grey,
        'frontiers': {name: frontiers[name]
                      for name in ('original', 'elapsed', 'overlapped', 'combined')},
    }
    data = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    data = data.replace('</', '<\\/')
    html = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pareto frontier</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:#18212f;font:14px system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1500px;margin:24px auto;padding:0 20px}h1{font-size:20px;font-weight:600;margin:0 0 4px}.sub{color:#687386;margin-bottom:14px}
.card{background:white;border:1px solid #dce2ea;border-radius:10px;padding:12px;box-shadow:0 2px 8px #18212f0b}svg{width:100%;height:min(72vh,760px);min-height:430px;display:block;overflow:visible}
.legend{display:flex;flex-wrap:wrap;gap:18px;padding:2px 12px 10px;color:#455164}.key{display:flex;align-items:center;gap:7px}.swatch{width:18px;height:3px;background:var(--c)}.dot{width:9px;height:9px;border-radius:50%;background:var(--c)}
.note{font-size:12px;color:#687386;padding:0 12px 5px}.tooltip{position:fixed;z-index:5;display:none;pointer-events:none;max-width:min(520px,calc(100vw - 24px));white-space:pre-wrap;overflow-wrap:anywhere;background:#111827;color:white;padding:10px 12px;border-radius:7px;box-shadow:0 5px 22px #0003;font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace}
.axis{stroke:#778295;stroke-width:1}.grid{stroke:#e7ebf0;stroke-width:1}.tick{fill:#5a6575;font-size:12px}.label{fill:#283446;font-size:13px}.point{cursor:crosshair}.point:hover{stroke:#111827;stroke-width:2.5}
</style></head><body><main><h1 id="title"></h1><div class="sub">Hover over a frontier point to inspect its case and configuration.</div><div class="card"><div class="legend" id="legend"></div><svg id="chart" role="img"></svg><div class="note">Grey points are candidates outside their own timing frontier. Curves may overlap exactly.</div></div></main><div class="tooltip" id="tip"></div>
<script id="plot-data" type="application/json">__DATA__</script><script>
(()=>{
const d=JSON.parse(document.getElementById('plot-data').textContent),svg=document.getElementById('chart'),tip=document.getElementById('tip'),NS='http://www.w3.org/2000/svg';
document.getElementById('title').textContent=d.title;svg.setAttribute('viewBox','0 0 1000 640');
const series=[['original','#0072B2','circle'],['elapsed','#E69F00','rect'],['overlapped','#009E73','triangle'],['combined','#CC79A7','diamond']];
const legend=document.getElementById('legend');for(const [n,c] of series){const item=document.createElement('span');item.className='key';item.innerHTML='<i class="swatch" style="--c:'+c+'"></i>'+n+' ('+d.frontiers[n].length+')';legend.appendChild(item)}
const item=document.createElement('span');item.className='key';item.innerHTML='<i class="dot" style="--c:#888"></i>Non-frontier candidates ('+d.candidates.length+')';legend.appendChild(item);
const W=1000,H=640,m={l:88,r:24,t:28,b:72},pw=W-m.l-m.r,ph=H-m.t-m.b;
let all=d.candidates.slice();for(const [n] of series)all=all.concat(d.frontiers[n]);
const tr=v=>d.scale==='log'?Math.log2(v):v, xs=all.map(p=>tr(p.x)),ys=all.map(p=>tr(p.y));
let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);if(xmin===xmax){xmin-=1;xmax+=1}if(ymin===ymax){ymin-=1;ymax+=1}
xmin=Math.min(0,xmin);ymin=Math.min(0,ymin);const xp=v=>m.l+(tr(v)-xmin)/(xmax-xmin)*pw,yp=v=>m.t+ph-(tr(v)-ymin)/(ymax-ymin)*ph;
function el(tag,attrs={},parent=svg){const e=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);parent.appendChild(e);return e}
function num(v){return Number(v).toLocaleString(undefined,{maximumSignificantDigits:4})}
function show(e,p,timing){const fields=[['case_id',p.case_id],['timing',timing],['x: tokens/s/user',p.x],['y: tokens/s/GPU',p.y],['batch_size',p.batch_size],['cp',p.cp],['dp',p.dp],['ep',p.ep],['pp_partition',p.pp_partition],['n_gpus',p.n_gpus],['hardware',p.hardware],['workload',p.workload]];tip.textContent=fields.filter(x=>x[1]!==undefined&&x[1]!==null).map(([k,v])=>k+': '+(typeof v==='number'?num(v):Array.isArray(v)?JSON.stringify(v):v)).join('\n');tip.style.display='block';move(e)}
function move(e){tip.style.left=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-12)+'px';tip.style.top=Math.min(e.clientY+14,innerHeight-tip.offsetHeight-12)+'px'}
function mark(p,c,shape,timing){const x=xp(p.x),y=yp(p.y),attrs={class:'point',fill:c,stroke:'white','stroke-width':1.2};let q;
 if(shape==='circle')q=el('circle',{...attrs,cx:x,cy:y,r:5});else if(shape==='rect')q=el('rect',{...attrs,x:x-4.5,y:y-4.5,width:9,height:9,rx:1});else if(shape==='triangle')q=el('path',{...attrs,d:`M ${x} ${y-6} L ${x+5.5} ${y+4.5} L ${x-5.5} ${y+4.5} Z`});else q=el('path',{...attrs,d:`M ${x} ${y-6} L ${x+6} ${y} L ${x} ${y+6} L ${x-6} ${y} Z`,fill:'white',stroke:c,'stroke-width':2});
 q.addEventListener('mouseenter',e=>show(e,p,timing));q.addEventListener('mousemove',move);q.addEventListener('mouseleave',()=>tip.style.display='none')}
for(let i=0;i<=5;i++){const f=i/5,x=m.l+pw*f,y=m.t+ph-ph*f;el('line',{class:'grid',x1:x,y1:m.t,x2:x,y2:m.t+ph});el('line',{class:'grid',x1:m.l,y1:y,x2:m.l+pw,y2:y});const xv=d.scale==='log'?Math.pow(2,xmin+(xmax-xmin)*f):Math.max(0,xmin+(xmax-xmin)*f),yv=d.scale==='log'?Math.pow(2,ymin+(ymax-ymin)*f):Math.max(0,ymin+(ymax-ymin)*f);el('text',{class:'tick',x,y:m.t+ph+22,'text-anchor':'middle'}).textContent=num(xv);el('text',{class:'tick',x:m.l-10,y:y+4,'text-anchor':'end'}).textContent=num(yv)}
el('line',{class:'axis',x1:m.l,y1:m.t+ph,x2:m.l+pw,y2:m.t+ph});el('line',{class:'axis',x1:m.l,y1:m.t,x2:m.l,y2:m.t+ph});el('text',{class:'label',x:m.l+pw/2,y:H-20,'text-anchor':'middle'}).textContent='Interactivity (tokens/s/user)';const yl=el('text',{class:'label',x:20,y:m.t+ph/2,'text-anchor':'middle',transform:`rotate(-90 20 ${m.t+ph/2})`});yl.textContent='Throughput (tokens/s/GPU)';
for(const p of d.candidates)el('circle',{class:'point',cx:xp(p.x),cy:yp(p.y),r:2.5,fill:'#888','fill-opacity':.35});
for(const [n,c,shape]of series){const ps=d.frontiers[n];if(ps.length){el('polyline',{points:ps.map(p=>xp(p.x)+','+yp(p.y)).join(' '),fill:'none',stroke:c,'stroke-width':n==='combined'?2.5:2,'stroke-dasharray':n==='combined'?'6 4':''});for(const p of ps)mark(p,c,shape,n)}}
})();</script></body></html>'''.replace('__DATA__', data)
    (out / f'{stem}.html').write_text(html, encoding='utf-8')


def plot_group(key, records, out, scale):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import EngFormatter
    candidates, frontiers = prepare(records)
    fig, ax = plt.subplots(figsize=(9, 6))
    # Grey means not retained on the candidate's own timing frontier.
    grey = []
    for timing, points in candidates.items():
        kept = {p['case_id'] for p in frontiers[timing]}
        grey.extend(p for p in points if p['case_id'] not in kept)
    ax.scatter([p['x'] for p in grey], [p['y'] for p in grey], s=9,
               color='#888888', alpha=.35, edgecolors='none',
               label=f'Non-frontier candidates ({len(grey)})', zorder=1)
    styles = {'original': ('#0072B2', 'o', '-'), 'elapsed': ('#E69F00', 's', '-'),
              'overlapped': ('#009E73', '^', '-'), 'combined': ('#CC79A7', 'D', '--')}
    for name, (color, marker, line) in styles.items():
        points = frontiers[name]
        ax.plot([p['x'] for p in points], [p['y'] for p in points],
                color=color, marker=marker, linestyle=line,
                markersize=7 if name == 'combined' else 4,
                markerfacecolor='none' if name == 'combined' else color,
                linewidth=1.6, label=f'{name} ({len(points)})', zorder=3)
    model, workload, hardware, gpus = key
    ax.set_title(f'{model} | {workload} | {hardware.upper()} | {gpus} GPUs')
    ax.set_xlabel('Interactivity (tokens/s/user)')
    ax.set_ylabel('Throughput (tokens/s/GPU)')
    if scale == 'log':
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
    else:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(EngFormatter(sep=''))
    ax.yaxis.set_major_formatter(EngFormatter(sep=''))
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    if not any(candidates.values()):
        ax.text(.5, .5, 'No feasible successful candidates', transform=ax.transAxes, ha='center')
    fig.text(.5, .02, 'Combined is the frontier of all three projections; coincident lines may overlap.\n'
             'Grey: dominated within its own timing. Infeasible projections are excluded.',
             ha='center', fontsize=8)
    stem = re.sub(r'[^a-zA-Z0-9_.-]', '_', f'{model}_{workload}_{hardware}_g{gpus}')
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'{stem}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    write_interactive_html(key, candidates, frontiers, out, stem, scale)
    rows = []
    union_ids = {(p['case_id'], p['timing']) for p in frontiers['combined']}
    for timing, points in candidates.items():
        own_ids = {p['case_id'] for p in frontiers[timing]}
        for p in points:
            rows.append(dict(model=model, workload=workload, hardware=hardware, n_gpus=gpus,
                             case_id=p['case_id'], timing=timing, x=p['x'], y=p['y'],
                             own_frontier=p['case_id'] in own_ids,
                             combined_frontier=(p['case_id'], timing) in union_ids,
                             batch_size=p.get('batch_size'), cp=p.get('cp'), dp=p.get('dp'),
                             ep=p.get('ep'), pp_partition=json.dumps(p.get('pp_partition'))))
    if rows:
        with (out / f'{stem}.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return dict(group=list(key), records=len(records),
                candidates={name: len(points) for name, points in candidates.items()},
                frontiers={name: len(points) for name, points in frontiers.items()},
                grey_points=len(grey), file=stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path,
                        help='JSONL file, model result directory, or parent containing model directories')
    parser.add_argument('--output-dir', type=Path, help='Must be a new directory')
    parser.add_argument('--workloads', nargs='+')
    parser.add_argument('--hardware', nargs='+')
    parser.add_argument('--gpu-counts', type=int, nargs='+')
    parser.add_argument('--axis-scale', choices=['linear', 'log'], default='linear')
    args = parser.parse_args()
    groups, paths = read_groups(args.input.resolve())
    groups = {k: v for k, v in groups.items()
              if (not args.workloads or k[1] in args.workloads)
              and (not args.hardware or k[2] in args.hardware)
              and (not args.gpu_counts or k[3] in args.gpu_counts)}
    if not groups:
        parser.error('No matching groups')
    out = (args.output_dir or ROOT / 'results' / ('pareto_timings_' +
           datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('MPLCONFIGDIR', str(out / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    summaries = [plot_group(key, records, out, args.axis_scale)
                 for key, records in sorted(groups.items())]
    (out / 'summary.json').write_text(json.dumps(dict(inputs=list(map(str, paths)),
        axis_scale=args.axis_scale, groups=summaries), indent=2) + '\n')
    print(f'Wrote {len(summaries)} interactive HTML plots, PNG/PDF figures, and candidate CSVs to {out}')


if __name__ == '__main__':
    main()
