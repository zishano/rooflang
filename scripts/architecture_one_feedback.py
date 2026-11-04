"""Trace-guided PP-boundary pilot. No source changes, API calls or metric changes."""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace, asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
from unittest.mock import patch

from rooflang.programs.experiments import find_pareto_frontier as search
from rooflang.programs.experiments.validate_architecture_one import validated_cases
from rooflang.runtime.simulator import Simulator


def stage_profile(sim, result, case):
    """Reconstruct spans with the same output-path clipping as the upstream metric."""
    _, included = search._output_path_kernels(sim._graph)
    gpus = search._ordered_gpus(sim._hardware)
    per_gpu = {}
    for gpu in gpus:
        entries = [e for e in result.trace if e.device is gpu and e.kernel in included
                   and e.end_us > result.measurement_start_us
                   and e.end_us > max(e.start_us, result.measurement_start_us)]
        if not entries:
            raise ValueError(f'No measured output-path work on {gpu.name}')
        start = min(max(e.start_us,result.measurement_start_us) for e in entries)
        end = max(e.end_us for e in entries)
        local = network = 0.
        kinds = defaultdict(lambda: {'device_events':0,'local_us':0.,'exposed_network_us':0.})
        for e in entries:
            s = max(e.start_us,result.measurement_start_us)
            l = max(0.,min(e.end_us,e.start_us+e.local_elapsed_time_us)-s)
            n = max(0.,min(e.end_us,e.start_us+e.network_elapsed_time_us)-s)
            exposed = max(0.,n-l)
            local += l
            network += exposed
            k = kinds[type(e.kernel).__name__]
            k['device_events'] += 1
            k['local_us'] += l
            k['exposed_network_us'] += exposed
        memory = sim._hardware.find_local_memory(gpu)
        projected = search._project_memory(result,len(case.pp_partition))
        per_gpu[gpu.name] = dict(start_us=start,end_us=end,span_us=end-start,
            local_us=local,exposed_network_us=network,by_operator_type=dict(kinds),
            memory=memory.name,peak_hbm_gb=result.peak_memory.get(memory,0)/1e9,
            projected_hbm_gb=projected.get(memory,0)/1e9,capacity_gb=memory.capacity_gb)
    stages = []
    for i,layers in enumerate(case.pp_partition):
        names = [g.name for g in gpus[i*case.ep:(i+1)*case.ep]]
        rows = [per_gpu[n] for n in names]
        span = max(r['end_us'] for r in rows)-min(r['start_us'] for r in rows)
        capacity = span * len(names)
        local = sum(r['local_us'] for r in rows)
        network = sum(r['exposed_network_us'] for r in rows)
        stages.append(dict(stage=i,layers=layers,devices=names,span_us=span,
            local_gpu_us=local,exposed_network_gpu_us=network,
            idle_gpu_us=max(0.,capacity-local-network),
            peak_hbm_gb=max(r['peak_hbm_gb'] for r in rows),
            projected_hbm_gb=max(r['projected_hbm_gb'] for r in rows)))
    return dict(stages=stages,gpus=per_gpu,
                bottleneck_stage=max(range(len(stages)),key=lambda i:stages[i]['span_us']))


def profiled_case(case):
    """Intercept only inside an isolated worker; run_case remains the evaluator."""
    search._select_model('dsv4_flash')
    captured = {}
    class ObservedSimulator(Simulator):
        def run(self):
            self._graph.validate()
            if any(self._placement.get_kernel_device(k).stream != 0
                   for k in self._placement.placed_kernels):
                raise ValueError('Unexpected non-default stream')
            result = super().run()
            captured.update(stage_profile(self,result,case))
            return result
    with patch.object(search,'Simulator',ObservedSimulator):
        record = search.run_case(case)
    record['model'] = 'dsv4_flash'
    if record['status'] == 'ok':
        reconstructed = max(s['span_us'] for s in captured['stages'])/1000
        if not math.isclose(reconstructed,record['total_gpu_elapsed_ms']/case.n_gpus,
                            rel_tol=1e-10,abs_tol=1e-9):
            raise AssertionError('Stage profile disagrees with upstream cycle metric')
    return record,captured


def propose(partition, profile):
    """First bounded action: move one boundary layer away from the slowest stage."""
    if len(partition) != 2:
        raise ValueError('This first controller supports PP=2 only')
    slow = profile['bottleneck_stage']
    if partition[slow] <= 1:
        return None
    p = list(partition)
    p[slow] -= 1
    p[1-slow] += 1
    return tuple(p)


def decision(record, baseline, incumbent):
    if record.get('status') != 'ok':
        return False,'simulation_failed'
    if not record.get('memory_feasible_elapsed',False):
        return False,'elapsed_memory_infeasible'
    # Fixed floating-point tolerance, not a performance-regression allowance.
    if record['latency_ms'] > baseline['latency_ms'] + max(1e-9,abs(baseline['latency_ms'])*1e-10):
        return False,'original_latency_regression'
    if record['tokens_per_s_gpu_elapsed'] <= incumbent['tokens_per_s_gpu_elapsed']*(1+1e-9):
        return False,'no_elapsed_throughput_improvement'
    return True,'accepted'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches',nargs='+',type=int,default=[8,32])
    p.add_argument('--budget',type=int,default=3,help='Maximum simulations per batch, including baseline')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.budget<1 or any(b<=0 or b%4 for b in a.batches) or len(set(a.batches)) != len(a.batches):
        p.error('budget >= 1; batches must be distinct positive multiples of 4')
    seed=validated_cases()[0]
    plan=dict(model='dsv4_flash',hardware='h200',gpu_count=8,context=8192,
              cp=1,dp=4,ep=4,pp=2,baseline=[22,21],batches=a.batches,budget_per_batch=a.budget,
              policy='move one layer away from measured slowest stage; no revisits; stop on rejected action',
              agent_api_used=False)
    files=[Path(__file__),Path(search.__file__),Path(Simulator.run.__code__.co_filename)]
    plan['source_sha256']={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    if a.dry_run:
        print(json.dumps(plan,indent=2));return
    out=a.output_dir or Path(__file__).resolve().parents[2]/'results/architecture_one'/datetime.now(timezone.utc).strftime('feedback_%Y%m%dT%H%M%S.%fZ')
    out.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    print(f'Output: {out.resolve()}',flush=True)
    summaries=[]
    with (out/'ledger.jsonl').open('x') as log, ProcessPoolExecutor(
        max_workers=1,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
        for batch in a.batches:
            current=replace(seed,batch_size=batch)
            visited=set();baseline=incumbent=None;stop='budget_exhausted'
            for index in range(a.budget):
                visited.add(current.pp_partition)
                print(f'batch={batch} experiment={index+1}: {current.case_id}',flush=True)
                record,profile=pool.submit(profiled_case,current).result()
                accepted=False;reason='baseline'
                if baseline is None:
                    baseline=record
                    if record.get('status')=='ok' and record.get('memory_feasible_elapsed',False):
                        incumbent=record;accepted=True
                    else:
                        reason='baseline_failed_or_infeasible'
                else:
                    accepted,reason=decision(record,baseline,incumbent)
                    if accepted:incumbent=record
                proposal=propose(current.pp_partition,profile) if accepted else None
                entry=dict(batch=batch,experiment=index+1,case=asdict(current),record=record,
                    profile=profile,accepted=accepted,reason=reason,next_proposal=proposal)
                log.write(json.dumps(entry)+'\n');log.flush()
                (out/f'b{batch}_experiment{index+1}.json').write_text(json.dumps(entry,indent=2))
                print(f'  {reason}; next={proposal}',flush=True)
                if not accepted:stop=reason;break
                if proposal is None or proposal in visited:
                    stop='no_unvisited_directional_neighbor';break
                current=replace(current,pp_partition=proposal)
            summaries.append(dict(batch=batch,evaluations=len(visited),stop=stop,baseline=baseline,best=incumbent,
                gain_pct=100*(incumbent['tokens_per_s_gpu_elapsed']/baseline['tokens_per_s_gpu_elapsed']-1) if incumbent else None))
    for f in files:
        if hashlib.sha256(f.read_bytes()).hexdigest()!=plan['source_sha256'][str(f.resolve())]:
            raise RuntimeError(f'Source changed during run: {f}')
    (out/'summary.json').write_text(json.dumps(summaries,indent=2))
    lines=['# 架构一：阶段反馈闭环实验','',
           '确定性规则控制器，无 LLM API 调用。elapsed 是单批次模拟推导的稳态指标。','',
           '| batch | 实验数 | 最佳分层 | 稳态吞吐收益 % | 停止原因 |','|---|---|---|---|---|']
    for s in summaries:
        lines.append(f'| {s["batch"]} | {s["evaluations"]} | {s["best"]["pp_partition"] if s["best"] else "无"} | {s["gain_pct"]} | {s["stop"]} |')
    lines += ['', '每次阶段跨度、按 GPU 的本地路径/暴露通信、逐卡 HBM 峰值见 b*_experiment*.json。',
              '接受条件：elapsed 内存可行、原始延迟不退步（仅允许浮点误差）、elapsed 吞吐增加。',
              '仅探索瓶颈方向的一层邻域，遇回访或拒绝即停止；不是全局最优证明，也不是完整 Agent 系统。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(out/'report.md',flush=True)
    if any(s['best'] is None for s in summaries):
        raise SystemExit(1)


if __name__=='__main__':main()
