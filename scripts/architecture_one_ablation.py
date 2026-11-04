"""Isolated add-one/leave-one-out ablation of Flash projection sharding."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace, asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

from rooflang.programs.experiments import find_pareto_frontier as search
from rooflang.programs.experiments.validate_architecture_one import validated_cases
from rooflang.programs.experiments.architecture_one_feedback import stage_profile
from rooflang.runtime.simulator import Simulator
from rooflang.language.kernels.comm import CommKernel

FAMILIES = {
    'endpoints': ('embedding','logits'),
    'attention': ('query_a','query_b','kv','output'),
    'moe_aux': ('gate','shared'),
}
ALL = tuple(g for groups in FAMILIES.values() for g in groups)


def variants():
    return [('legacy',None),('none',()),('full',ALL)] + [
        ('only_'+name,groups) for name,groups in FAMILIES.items()] + [
        ('without_'+name,tuple(g for g in ALL if g not in groups))
        for name,groups in FAMILIES.items()]


def inspect_result(sim, result, case):
    profile = stage_profile(sim,result,case)
    unique = {e.kernel for e in result.trace if e.end_us > result.measurement_start_us}
    comm = [k for k in unique if isinstance(k,CommKernel)]
    noncomm = [k for k in unique if not isinstance(k,CommKernel)]
    by_type = defaultdict(float)
    for k in noncomm:
        by_type[type(k).__name__] += float(k.flops)
    metrics = dict(
        measured_unique_kernels=len(unique),
        measured_loaded_weight_bytes=sum(float(k.loaded_weight_bytes) for k in unique),
        measured_comm_kernels=len(comm),
        measured_comm_types=dict(Counter(type(k).__name__ for k in comm)),
        # A modeled collective byte estimate, not physical link traffic summed over routes.
        modeled_collective_transferred_bytes=sum(float(getattr(k,'transferred_bytes',0)) for k in comm),
        measured_noncomm_flops_by_type=dict(by_type),
        measured_reduction_flops=sum(float(k.flops) for k in comm),
        local_gpu_us=sum(s['local_gpu_us'] for s in profile['stages']),
        exposed_network_gpu_us=sum(s['exposed_network_gpu_us'] for s in profile['stages']),
        stages=profile['stages'])
    return metrics


def evaluate(case, name, groups, legacy_path):
    search._select_model('dsv4_flash')
    if name == 'legacy':
        spec=importlib.util.spec_from_file_location('rooflang_ablation_legacy',legacy_path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        optimize=module.optimize_model_cluster_decode
    else:
        original=search.optimize_model_cluster_decode
        def optimize(*args,**kwargs):
            return original(*args,projection_groups=list(groups),**kwargs)
    captured={}
    class Observed(Simulator):
        def run(self):
            self._graph.validate()
            self._placement.validate(self._graph)
            assert all(self._placement.get_kernel_device(k).stream==0 for k in self._placement.placed_kernels)
            result=super().run()
            captured.update(inspect_result(self,result,case))
            return result
    with patch.object(search,'optimize_model_cluster_decode',optimize),patch.object(search,'Simulator',Observed):
        record=search.run_case(case)
    if record['status']=='ok':
        assert math.isclose(max(s['span_us'] for s in captured['stages'])/1000,
                            record['total_gpu_elapsed_ms']/case.n_gpus,rel_tol=1e-10,abs_tol=1e-9)
    return dict(variant=name,groups=groups,record=record,mechanism=captured)


def write_reports(out, entries):
    import csv
    fields=['batch','variant','status','latency_ms','elapsed_gpu','overlapped_gpu','hbm_gb',
            'elapsed_feasible','gain_vs_legacy_pct','gain_vs_none_pct','gain_vs_full_pct',
            'loaded_weights_GB','comm_kernels','collective_GB','local_gpu_us','exposed_network_gpu_us']
    rows=[]
    for entry in entries:
        r=entry['record'];m=entry['mechanism'];batch=r['batch_size']
        row=dict(batch=batch,variant=entry['variant'],status=r['status'])
        if r['status']=='ok':
            row.update(latency_ms=r['latency_ms'],elapsed_gpu=r['tokens_per_s_gpu_elapsed'],
                       overlapped_gpu=r['tokens_per_s_gpu_overlapped'],hbm_gb=r['peak_hbm_gb_elapsed'],
                       elapsed_feasible=r['memory_feasible_elapsed'],
                       loaded_weights_GB=m['measured_loaded_weight_bytes']/1e9,
                       comm_kernels=m['measured_comm_kernels'],
                       collective_GB=m['modeled_collective_transferred_bytes']/1e9,
                       local_gpu_us=m['local_gpu_us'],exposed_network_gpu_us=m['exposed_network_gpu_us'])
            for base in ('legacy','none','full'):
                match=next((e for e in entries if e['variant']==base and e['record']['batch_size']==batch),None)
                if match and match['record']['status']=='ok':
                    row['gain_vs_'+base+'_pct']=100*(r['tokens_per_s_gpu_elapsed']/match['record']['tokens_per_s_gpu_elapsed']-1)
        rows.append(row)
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    lines=['# 架构一：投影分片消融','',
           '固定 Flash / 8×H200 / 8K / CP1 DP4 EP4 PP2 [22,21]；仅改变优化组。每个 batch 独立比较。',
           'legacy 是运行前 Git HEAD 中的优化器；none 是最新优化器禁用所有投影组，仍保留 CP=1 的结构简化。',
           'full 显式开启八个组；本轮 batch≤128 时与最新自动策略等价。',
           '仅输出层/共享专家上下游成对分片；不将耦合变换强拆。','',
           '| batch | 方案 | 状态 | TPOT ms | elapsed tok/s/GPU | 比旧版 % | 比全开 % | HBM GB | 读取权重 GB/步 | 暴露网络 GPU·μs |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        cols=[str(row['batch']),row['variant'],row['status']]+[
            f'{row[k]:.6f}' if k in row else '—' for k in
            ['latency_ms','elapsed_gpu','gain_vs_legacy_pct','gain_vs_full_pct','hbm_gb','loaded_weights_GB','exposed_network_gpu_us']]
        lines.append('| '+' | '.join(cols)+' |')
    lines+=['','## 指标与结论边界','',
            'elapsed 吞吐仅在对应内存可行时有意义；原始延迟、elapsed、overlapped 分别记录，不合并。',
            '权重读取按测量窗口中的唯一算子累计，通信多设备 trace 不重复计数；不等于驻留权重容量。',
            'collective 字节数是算子的 transferred_bytes 模型估计，不是各物理链路流量总和。local 与 exposed network 是累计 GPU 时间，不能当单步延迟相加。',
            '独立收益和留一收益不能直接相加，跨算子的通信消除及布局转换会产生相互作用。',
            '图/端口/放置校验通过不代表浮点数值等价已证明；未执行真实模型推理。本实验不调用 Agent API。',
            '详细分组、原始结果和阶段机制数据见 plan.json、raw_results.jsonl。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def isolated_run(out):
    assert Path(__file__).resolve().is_relative_to(out/'source'), 'Must run inside frozen copy'
    plan=json.loads((out/'plan.json').read_text())
    seed=validated_cases()[0]
    entries=[]
    with (out/'raw_results.jsonl').open('x') as f,ProcessPoolExecutor(
        max_workers=1,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
        for batch in plan['batches']:
            for name,groups in variants():
                print(f'[{len(entries)+1}/{len(plan["batches"])*9}] batch={batch} {name}',flush=True)
                entry=pool.submit(evaluate,replace(seed,batch_size=batch),name,groups,str(out/'legacy_optimization.py')).result()
                entries.append(entry);f.write(json.dumps(entry)+'\n');f.flush()
                (out/f'b{batch}_{name}.json').write_text(json.dumps(entry,indent=2))
                r=entry['record'];print(f'  {r["status"]}: elapsed={r.get("tokens_per_s_gpu_elapsed")} latency={r.get("latency_ms")}',flush=True)
                write_reports(out,entries)
    return int(any(e['record']['status']!='ok' for e in entries))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches',type=int,nargs='+',default=[8,32,128])
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--isolated-run',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.isolated_run:
        raise SystemExit(isolated_run(a.isolated_run.resolve()))
    if any(b<=0 or b%4 or b>256 for b in a.batches) or len(set(a.batches))!=len(a.batches):
        p.error('Distinct batches must be positive multiples of 4, <=256 (full policy region)')
    seed=validated_cases()[0]
    root=Path(__file__).resolve().parents[2]
    legacy=subprocess.check_output(['git','show','HEAD:programs/models/dsv4_flash/optimization.py'],cwd=root)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if b'projection_groups' in legacy:
        p.error('Git HEAD is no longer pre-sharding; select/verify a legacy version before running')
    plan=dict(batches=a.batches,case=asdict(seed),families=FAMILIES,variants=variants(),
              legacy_commit=commit,legacy_sha256=hashlib.sha256(legacy).hexdigest(),
              simulations=len(a.batches)*len(variants()),serial=True)
    if a.dry_run:
        print(json.dumps(plan,indent=2));return
    out=(a.output_dir or root/'results/architecture_one'/datetime.now(timezone.utc).strftime('ablation_%Y%m%dT%H%M%S.%fZ')).resolve()
    out.mkdir(parents=True,exist_ok=False)
    source=out/'source'/'rooflang';source.mkdir(parents=True)
    paths=[root/'__init__.py']
    for d in ['language','runtime','programs']:
        paths+=list((root/d).rglob('*.py'))
    before={}
    for src in paths:
        relative=src.relative_to(root);data=src.read_bytes();dest=source/relative
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        before[str(relative)]=hashlib.sha256(data).hexdigest()
    plan['source_sha256']=before
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    (out/'legacy_optimization.py').write_bytes(legacy)
    env=dict(os.environ,PYTHONPATH=str(out/'source'),PYTHONDONTWRITEBYTECODE='1')
    cmd=[sys.executable,'-B','-m','rooflang.programs.experiments.architecture_one_ablation','--isolated-run',str(out)]
    print(f'Isolated experiment: {out}',flush=True)
    with (out/'run.log').open('x') as log:
        proc=subprocess.Popen(cmd,cwd=out/'source',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in proc.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        status=proc.wait()
    original_changes=[p for p,h in before.items() if hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]
    snapshot_changes=[p for p,h in before.items() if hashlib.sha256((source/p).read_bytes()).hexdigest()!=h]
    (out/'integrity.json').write_text(json.dumps(dict(original_changes=original_changes,snapshot_changes=snapshot_changes,returncode=status),indent=2))
    print(out/'report.md',flush=True)
    raise SystemExit(status or int(bool(original_changes or snapshot_changes)))


if __name__=='__main__':main()
