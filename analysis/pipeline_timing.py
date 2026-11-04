"""Independent pipeline timing report matching find_pareto_frontier semantics.

No dependency on the single-layer reporting scripts or their artifacts.
Original is one-batch output latency. Elapsed and overlapped are cycle estimates,
not event durations or measured overlap intervals.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from rooflang.language.hardware.component import Compute
from rooflang.programs.models import load_model
from rooflang.programs.presets.h200 import H200Cluster
from rooflang.programs.presets.b300 import B300Cluster
from rooflang.programs.experiments.find_pareto_frontier import (
    _output_path_kernels, _gpu_timing_totals_us, _gpu_timing_metrics,
)
from rooflang.runtime.simulator import Simulator


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model', choices=['dsv4_flash', 'dsv4_pro'], default='dsv4_flash')
    p.add_argument('--hardware', choices=['H200Cluster1Node', 'B300Cluster1Node'], default='H200Cluster1Node')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--context', type=int, default=8192)
    p.add_argument('--cp', type=int, default=2)
    p.add_argument('--dp', type=int, default=2)
    p.add_argument('--ep', type=int, default=4)
    p.add_argument('--pp-partition', type=int, nargs='+', default=None)
    p.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1]/'results/pipeline_timing')
    p.add_argument('--print-config', action='store_true', help='只检查并打印参数，不模拟')
    a = p.parse_args()
    model = load_model(a.model)
    a.pp_partition = a.pp_partition or [model.N_LAYERS]
    if min(a.batch_size, a.context, a.cp, a.dp, a.ep, *a.pp_partition) <= 0:
        p.error('batch/context/并行度/各段层数必须为正整数')
    if sum(a.pp_partition) != model.N_LAYERS:
        p.error(f'pp-partition 的总和必须等于 {model.N_LAYERS}')
    if a.cp*a.dp != a.ep or a.ep*len(a.pp_partition) != 8:
        p.error('要求 cp*dp == ep 且 ep*PP段数 == 8')
    if a.batch_size % (a.cp*a.dp) or model.N_EXPERTS % a.ep:
        p.error('batch-size 必须能被 cp*dp 整除，专家数必须能被 ep 整除')
    if a.context % a.cp or model.WINDOW % a.cp:
        p.error('context 和 WINDOW 必须能被 cp 整除')
    for ratio in set(model.COMPRESS_RATIOS)-{0}:
        if a.context % (ratio*a.cp):
            p.error(f'context 必须能被 {ratio}*cp 整除')
    return a, model


def merged(intervals):
    out=[]
    for start,end in sorted(intervals):
        if end <= start:
            continue
        if out and start <= out[-1][1]+1e-9:
            out[-1]=(out[-1][0],max(end,out[-1][1]))
        else:
            out.append((start,end))
    return out


def collect(result, graph, gpus, stages):
    outputs, included = _output_path_kernels(graph)
    measurement = result.measurement_start_us
    output_end = max(e.end_us for e in result.trace if e.kernel in outputs)
    original = output_end-measurement
    devices = {}
    kernel_ids = {k: f'K{i:06d}' for i, k in enumerate(graph.topological_sort())}
    overlap_pairs = []
    network_records = []
    # Reproduce the public report's clipping and exposed-network accounting.
    for dev in gpus:
        entries = [e for e in result.trace if e.device is dev and e.kernel in included
                   and e.end_us > measurement]
        first = min(max(e.start_us,measurement) for e in entries)
        final = max(e.end_us for e in entries)
        local, network = [], []
        c=n=0.0
        compute_chunks=[]
        network_chunks=[]
        for e in entries:
            start,end = max(e.start_us,measurement),min(e.end_us,final)
            if end <= start:
                continue
            local_us=max(0.0,min(end,e.start_us+e.local_elapsed_time_us)-start)
            network_us=max(0.0,min(end,e.start_us+e.network_elapsed_time_us)-start)
            exposed=max(0.0,network_us-local_us)
            c+=local_us
            n+=exposed
            if local_us:
                local.append((start-measurement,start+local_us-measurement))
            if exposed:
                network.append((start+local_us-measurement,start+network_us-measurement))
                network_chunks.append(dict(kernel_id=kernel_ids[e.kernel],
                    operator=type(e.kernel).__name__, actual_start_us=start-measurement,
                    original_network_us=e.network_time_us, network_elapsed_us=network_us,
                    local_elapsed_us=local_us, exposed_us=exposed))
                network_records.append(dict(device=dev.name, **network_chunks[-1]))
            if local_us:
                compute_chunks.append(dict(kernel_id=kernel_ids[e.kernel],
                    operator=type(e.kernel).__name__, actual_start_us=start-measurement,
                    compute_us=local_us))
        span=final-first
        bubble=span-c-n
        if bubble < -1e-6:
            raise ValueError('GPU 的本地/暴露通信区间存在并发，无法用串行资源分解绘制当前图。')
        # Explicitly materialize the Pareto ideal-overlap budget. FIFO pairing
        # is a transparent attribution policy; the simulator does not schedule
        # these pairs or provide their ideal virtual timestamps.
        compute_chunks.sort(key=lambda x:(x['actual_start_us'],x['kernel_id']))
        network_chunks.sort(key=lambda x:(x['actual_start_us'],x['kernel_id']))
        ci=ni=0
        c_remaining=compute_chunks[0]['compute_us'] if compute_chunks else 0.0
        n_remaining=network_chunks[0]['exposed_us'] if network_chunks else 0.0
        # Start pair offsets at this GPU's elapsed offset from its PP-stage start.
        # Stage starts are assigned below after all GPU traces are collected.
        virtual_cursor=0.0
        while ci<len(compute_chunks) and ni<len(network_chunks):
            amount=min(c_remaining,n_remaining)
            if amount<=1e-12:
                if c_remaining<=1e-12:
                    ci+=1
                    c_remaining=compute_chunks[ci]['compute_us'] if ci<len(compute_chunks) else 0.0
                if n_remaining<=1e-12:
                    ni+=1
                    n_remaining=network_chunks[ni]['exposed_us'] if ni<len(network_chunks) else 0.0
                continue
            comp=compute_chunks[ci]; net=network_chunks[ni]
            pair=dict(device=dev.name,compute_kernel_id=comp['kernel_id'],
                compute_operator=comp['operator'],communication_kernel_id=net['kernel_id'],
                communication_operator=net['operator'],compute_actual_start_us=comp['actual_start_us'],
                communication_actual_start_us=net['actual_start_us'],
                overlap_us=amount,virtual_start_us=virtual_cursor,
                virtual_end_us=virtual_cursor+amount)
            overlap_pairs.append(pair)
            virtual_cursor+=amount
            c_remaining-=amount;n_remaining-=amount
            if c_remaining<=1e-12:
                ci+=1;c_remaining=compute_chunks[ci]['compute_us'] if ci<len(compute_chunks) else 0.0
            if n_remaining<=1e-12:
                ni+=1;n_remaining=network_chunks[ni]['exposed_us'] if ni<len(network_chunks) else 0.0
        devices[dev]=dict(device=dev.name,first_us=first-measurement,end_us=final-measurement,
                          elapsed_us=span,compute_us=c,exposed_communication_us=n,
                          bubble_us=max(0.0,bubble),hidden_us=min(c,n),
                          overlapped_us=span-min(c,n),local_intervals=merged(local),
                          network_intervals=merged(network))
    stage_rows=[]
    for index,group in enumerate(stages):
        first=min(devices[d]['first_us'] for d in group)
        end=max(devices[d]['end_us'] for d in group)
        ideal_end=max(devices[d]['first_us']+devices[d]['overlapped_us'] for d in group)
        stage_rows.append(dict(stage=index,devices=[d.name for d in group],
                               first_us=first,end_us=end,elapsed_us=end-first,
                               overlapped_end_us=ideal_end,overlapped_us=ideal_end-first))
    elapsed=max(s['elapsed_us'] for s in stage_rows)
    overlapped=max(s['overlapped_us'] for s in stage_rows)
    # Verify every reported aggregate against the existing implementation.
    reference=_gpu_timing_totals_us(result,len(gpus),included,stages)
    actual=(elapsed*len(gpus),sum(d['compute_us'] for d in devices.values()),
            sum(d['exposed_communication_us'] for d in devices.values()),
            overlapped*len(gpus),sum(d['elapsed_us'] for d in devices.values()))
    for got,want in zip(actual,reference):
        if not math.isclose(got,want,abs_tol=1e-6,rel_tol=1e-9):
            raise AssertionError(f'与 Pareto 定义不一致：{got} != {want}')
    # Shift per-GPU virtual pair intervals by the GPU's skew within its stage.
    for stage in stage_rows:
        stage_devices={d.name for d in stages[stage['stage']]}
        stage_first=stage['first_us']
        for pair in overlap_pairs:
            if pair['device'] in stage_devices:
                skew=devices[next(d for d in gpus if d.name==pair['device'])]['first_us']-stage_first
                pair['virtual_start_us']+=skew
                pair['virtual_end_us']+=skew
    communication_rows=[]
    for net in network_records:
        assigned=[p for p in overlap_pairs if p['device']==net['device'] and
                  p['communication_kernel_id']==net['kernel_id'] and
                  math.isclose(p['communication_actual_start_us'],net['actual_start_us'],abs_tol=1e-9)]
        communication_rows.append(dict(device=net['device'],kernel_id=net['kernel_id'],
            operator=net['operator'],actual_start_us=net['actual_start_us'],
            original_network_us=net['original_network_us'],network_elapsed_us=net['network_elapsed_us'],
            local_elapsed_us=net['local_elapsed_us'],exposed_communication_us=net['exposed_us'],
            hidden_us=sum(p['overlap_us'] for p in assigned),
            paired_compute_kernels='; '.join(dict.fromkeys(p['compute_kernel_id']+' '+p['compute_operator'] for p in assigned)),
            ideal_virtual_intervals='; '.join(f'[{p["virtual_start_us"]:.6f},{p["virtual_end_us"]:.6f}]' for p in assigned)))
    # The pair attribution must exactly account for each GPU's ideal budget.
    for dev in gpus:
        assigned=sum(p['overlap_us'] for p in overlap_pairs if p['device']==dev.name)
        if not math.isclose(assigned,devices[dev]['hidden_us'],abs_tol=1e-6,rel_tol=1e-9):
            raise AssertionError(f'{dev.name}: FIFO overlap assignment {assigned} != ideal hidden budget {devices[dev]["hidden_us"]}')
    return devices,stage_rows,dict(original_us=original,elapsed_cycle_us=elapsed,
        overlapped_cycle_us=overlapped,
        elapsed_user_latency_us=elapsed*len(stages),
        overlapped_user_latency_us=overlapped*len(stages),
        preload_us=measurement,post_output_us=result.total_time_us-output_end,
        reference_checked=True,
        overlap_policy='FIFO by simulated event start, per GPU; virtual placement is illustrative, aggregate hidden budget remains min(C,N)'),included,overlap_pairs,communication_rows


def plot(mode,devices,stage_rows,gpus,config,metrics,out):
    titles={'original':'Original: one-batch output completion',
            'elapsed':'Elapsed: bottleneck PP-stage cycle',
            'overlapped':'Overlapped: ideal dataflow-overlap cycle'}
    key={'original':'original_us','elapsed':'elapsed_cycle_us','overlapped':'overlapped_cycle_us'}[mode]
    duration=metrics[key]
    fig,ax=plt.subplots(figsize=(17,9))
    fig.suptitle(titles[mode],x=.12,y=.97,ha='left',fontsize=19,fontweight='bold')
    fig.text(.12,.92,f'{config["model"]} | {config["hardware"]} | batch {config["batch_size"]} | '
             f'CP {config["cp"]} / DP {config["dp"]} / EP {config["ep"]} / PP {config["pp_partition"]}',fontsize=11)
    fig.text(.12,.875,f'{"Output latency" if mode=="original" else "Stage cycle"}: {duration:.3f} us'
             +(f' | Pareto user-latency projection: {duration*len(stage_rows):.3f} us (cycle x PP)' if mode!='original' else ''),fontsize=12)
    ticks=[];labels=[]
    ccolor,ncolor='#2563eb','#f97316'
    for stage in stage_rows:
        stage_index=stage['stage']
        for d in gpus:
            if d.name not in stage['devices']:
                continue
            rank=gpus.index(d); row=rank*2.4
            info=devices[d]
            shift=0 if mode=='original' else stage['first_us']
            first=info['first_us']-shift
            end=(info['first_us']+info['overlapped_us'] if mode=='overlapped' else info['end_us'])-shift
            if mode=='overlapped':
                # This packing is a schematic for the aggregate formula. It
                # does not claim a realizable per-kernel dependency schedule.
                local=[(first,first+info['compute_us'])]
                network=[(first,first+info['exposed_communication_us'])]
            else:
                local=[(a-shift,b-shift) for a,b in info['local_intervals']]
                network=[(a-shift,b-shift) for a,b in info['network_intervals']]
            gaps=[]
            cursor=first
            for a,b in merged(local+network):
                if a>cursor+1e-8:
                    gaps.append((cursor,a-cursor))
                cursor=max(cursor,b)
            if end>cursor+1e-8:
                gaps.append((cursor,end-cursor))
            for y,intervals,color in [(row,local,ccolor),(row+.7,network,ncolor)]:
                ax.broken_barh([(first,end-first)],(y-.23,.46),facecolors='#f1f5f9')
                ax.broken_barh(gaps,(y-.23,.46),facecolors='#94a3b8')
                ax.broken_barh([(a,b-a) for a,b in intervals if b>a],(y-.23,.46),facecolors=color)
            ax.text(end+duration*.008,row+.35,f'{end-first:.2f} us',fontsize=8,va='center')
            ticks.append(row+.35);labels.append(f'PP {stage_index} / GPU {rank}\nlocal / exposed net')
        # All stage boundaries include the rank skew retained by the reference.
        top=min(gpus.index(d)*2.4 for d in gpus if d.name in stage['devices'])-.6
        bottom=max(gpus.index(d)*2.4 for d in gpus if d.name in stage['devices'])+1.1
        stage_end=stage['end_us'] if mode=='original' else stage['overlapped_us'] if mode=='overlapped' else stage['elapsed_us']
        ax.vlines(stage_end,top,bottom,color='#334155',linestyle=':',linewidth=1)
    ax.axvline(duration,color='#111827',linestyle='--',linewidth=1.2)
    ax.set_yticks(ticks,labels)
    ax.set_ylim(len(gpus)*2.4-.8,-.9)
    ax.set_xlim(0,duration*1.12)
    ax.set_xlabel('Time from measurement start (us)' if mode=='original' else 'Time from each PP stage start (us)',fontsize=11)
    ax.grid(axis='x',alpha=.18)
    for side in ('top','right','left'):ax.spines[side].set_visible(False)
    ax.tick_params(axis='y',length=0,labelsize=9)
    fig.legend(handles=[Patch(color=ccolor,label='Local compute + memory path (C)'),
                        Patch(color=ncolor,label='Exposed network tail (N)'),
                        Patch(color='#94a3b8',label='Dependency / scheduling bubble'),
                        Patch(color='#f1f5f9',label='Inactive on this lane')],
               loc='lower center',bbox_to_anchor=(.55,.09),ncol=4,frameon=False,fontsize=9)
    notes={
        'original':'Actual single-batch simulation on the output dependency path. KV preload excluded. No ideal overlap is introduced.',
        'elapsed':'Stage time = latest GPU completion - earliest GPU start; cycle = max(stage time). Stages are aligned here for comparison.',
        'overlapped':'Ideal schematic: each GPU T\' = T - min(C, N); stage cycle is recalculated after contraction. Bubble and rank skew are retained.',
    }
    fig.text(.12,.065,notes[mode],fontsize=10)
    fig.text(.12,.035,'Overlapped is an aggregate ideal estimate, not observed overlap or multi-batch/double-buffer simulation. '
             'Metrics follow find_pareto_frontier.py.',fontsize=9)
    fig.subplots_adjust(left=.12,right=.97,top=.82,bottom=.18)
    for suffix in ('png','svg'):fig.savefig(out/f'{mode}.{suffix}',dpi=170)
    plt.close(fig)


def plot_overlap_pairs(pairs, stages, gpus, config, out):
    """Show every FIFO-attributed compute/communication overlap assignment."""
    by_device={}
    for pair in pairs:
        by_device.setdefault(pair['device'],[]).append(pair)
    stage_by_device={d.name:i for i,group in enumerate(stages) for d in group}
    rank_by_device={d.name:i for i,d in enumerate(gpus)}
    with PdfPages(out/'overlap_pairs.pdf') as pdf:
        for device,records in by_device.items():
            records.sort(key=lambda p:(p['virtual_start_us'],p['compute_kernel_id'],p['communication_kernel_id']))
            for page_start in range(0,len(records),24):
                page=records[page_start:page_start+24]
                fig,ax=plt.subplots(figsize=(17,10))
                fig.suptitle(f'FIFO overlap attribution — PP {stage_by_device[device]} / GPU {rank_by_device[device]}',
                             x=.10,y=.97,ha='left',fontsize=17,fontweight='bold')
                fig.text(.10,.935,f'{config["model"]} | pair rows {page_start+1}–{page_start+len(page)} of {len(records)}',fontsize=10)
                for row,pair in enumerate(page):
                    y=len(page)-row-1
                    start,end=pair['virtual_start_us'],pair['virtual_end_us']
                    ax.broken_barh([(start,end-start)],(y-.28,.22),facecolors='#2563eb')
                    ax.broken_barh([(start,end-start)],(y+.04,.22),facecolors='#f97316')
                    ax.text(end,y-.17,f'{pair["compute_kernel_id"]} {pair["compute_operator"]}  (sim start {pair["compute_actual_start_us"]:.2f} μs)',
                            va='center',ha='left',fontsize=7.5)
                    ax.text(end,y+.15,f'{pair["communication_kernel_id"]} {pair["communication_operator"]}  (sim start {pair["communication_actual_start_us"]:.2f} μs)',
                            va='center',ha='left',fontsize=7.5)
                    ax.text(start,y+.48,f'{pair["overlap_us"]:.3f} μs',ha='left',va='center',fontsize=7)
                ax.set_yticks([])
                xmax=max(p['virtual_end_us'] for p in page)
                ax.set_xlim(0,max(xmax*1.35,1))
                ax.set_ylim(-.7,len(page)-.1)
                ax.set_xlabel('Illustrative ideal time from PP-stage start (μs)')
                ax.grid(axis='x',alpha=.2)
                for side in ('top','right','left'):ax.spines[side].set_visible(False)
                fig.legend(handles=[Patch(color='#2563eb',label='Compute portion assigned to overlap'),
                                    Patch(color='#f97316',label='Communication portion assigned to overlap')],
                           loc='lower center',bbox_to_anchor=(.55,.025),ncol=2,frameon=False)
                fig.text(.10,.025,'FIFO attribution of the ideal min(C,N) budget; paired bars are the same assigned interval. '
                         'Actual simulator event starts are printed beside each kernel; these virtual intervals are not observed schedule times.',fontsize=8)
                fig.subplots_adjust(left=.08,right=.98,top=.90,bottom=.12)
                pdf.savefig(fig)
                plt.close(fig)


def main():
    args,model=arguments()
    out=args.output_dir.resolve()
    config={k:v for k,v in vars(args).items() if k!='print_config'}
    config['output_dir']=str(out)
    config.update(stage='decode',measurement_start='read_input',
                  metric_source='programs/experiments/find_pareto_frontier.py:_gpu_timing_totals_us')
    print(json.dumps(config,ensure_ascii=False,indent=2),flush=True)
    if args.print_config:return
    hw=(H200Cluster if args.hardware=='H200Cluster1Node' else B300Cluster)(n_nodes=1)
    gpus=sorted([d for d in hw.nodes if isinstance(d,Compute) and d.kind=='gpu'],key=lambda d:int(d.name.rsplit('-',1)[1]))
    stages=tuple(tuple(gpus[i*args.ep:(i+1)*args.ep]) for i in range(len(args.pp_partition)))
    graph,layers,emb,read,kv,head=model.declare_model(seq_prefill=args.context,decode=True,batch_size=args.batch_size)
    graph,placement=model.optimize_model_cluster_decode(graph,layers,hw,emb,read,kv,head,
        seq_prefill=args.context,cp=args.cp,dp=args.dp,ep=args.ep,pp_partition=args.pp_partition,n_gpus=len(gpus))
    result=Simulator(graph,placement,hw,measurement_start=read).run()
    devices,stage_rows,metrics,included,overlap_pairs,communication_rows=collect(result,graph,gpus,stages)
    metrics['pareto_metrics']=_gpu_timing_metrics(result,args.batch_size,1,len(gpus),included,
                                               metrics['original_us'],stages)
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    data=dict(config=config,metrics=metrics,stages=stage_rows,gpus=list(devices.values()))
    (out/'timing.json').write_text(json.dumps(data,indent=2))
    for filename,records in [('stages.csv',stage_rows),('gpus.csv',[{k:v for k,v in d.items() if not k.endswith('_intervals')} for d in devices.values()])]:
        with (out/filename).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    for filename,records in [('overlap_pairs.csv',overlap_pairs),('communication_overlap.csv',communication_rows)]:
        with (out/filename).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]) if records else ['device']);writer.writeheader();writer.writerows(records)
    (out/'overlap_pairs.json').write_text(json.dumps(overlap_pairs,indent=2))
    for mode in ('original','elapsed','overlapped'):plot(mode,devices,stage_rows,gpus,config,metrics,out)
    plot_overlap_pairs(overlap_pairs,stages,gpus,config,out)
    report=['# Pipeline timing: original / elapsed / overlapped','',
            '三种指标均基于同一次单批 decode 仿真，已逐项与仓库 Pareto 函数校验。',
            f'- original：{metrics["original_us"]:.6f} μs，输出路径完成时间减 measurement_start。',
            f'- elapsed：{metrics["elapsed_cycle_us"]:.6f} μs，最慢 PP stage 的周期。',
            f'- overlapped：{metrics["overlapped_cycle_us"]:.6f} μs，理想隐藏后的最慢 PP stage 周期。','',
            '对每个 GPU：C 为资源竞争后的本地计算/访存路径之和；N 为 max(网络路径 − 本地路径, 0) 之和。',
            'GPU 活动跨度 T 为末事件结束减首事件开始。理想跨度 T\'=T−min(C,N)，保留 bubble。',
            'stage 时间取组内最晚结束减最早开始；overlapped 保留原始 GPU 起点，仅将终点改为 start+T\'。',
            'elapsed / overlapped 是周期，不等于单用户延迟。Pareto 的 user latency = 周期 × PP 段数。',
            'overlapped 汇总图遵循上述聚合公式。overlap_pairs.pdf / overlap_pairs.csv 将每一份 min(C,N) 隐藏预算按 GPU 内计算与通信事件的仿真开始顺序 FIFO 配对；同一对的两条矩形表示分配给重叠的时长。',
            '配对表同时记录模拟器中的事件开始时间和解释性虚拟区间。模拟器没有执行或记录这些虚拟配对，因而它们是明确的归因/展示策略，不是实际发生的 kernel overlap 时间；也不是多批 overlap/double buffering。communication_overlap.csv 按通信事件汇总其 exposed 时长、分配隐藏量和对应计算 kernel。',
            '输出路径范围与 `_output_path_kernels` 相同；包含终端依赖 Nop，排除无关尾部工作；不包含 KV preload。','',
            '![original](original.png)','![elapsed](elapsed.png)','![overlapped](overlapped.png)','',
            '详细参数：config.json；阶段：stages.csv；GPU：gpus.csv；指标与绘图区间：timing.json；逐对配对：overlap_pairs.csv / overlap_pairs.pdf；逐通信事件：communication_overlap.csv。','']
    (out/'README.md').write_text('\n'.join(report))
    print(json.dumps({k:v for k,v in metrics.items() if k!='pareto_metrics'},indent=2),flush=True)
    print(f'三个汇总图和逐对时间线已生成：{out}/{{original,elapsed,overlapped}}.png, {out}/overlap_pairs.pdf',flush=True)


if __name__=='__main__':main()
