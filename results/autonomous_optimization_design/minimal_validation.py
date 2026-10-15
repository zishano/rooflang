"""Executable scheduling prototype on a DSV4 Flash MoE fragment.

No tensor values or CUDA kernels execute here. Checks cover IR invariants,
DAG ordering, simulator contention and capacity accounting only.
Run from the RoofLang checkout with PYTHONPATH pointing to its parent.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import csv

from rooflang.language.graph import ComputeGraph
from rooflang.language.kernels.forward import TokenCombine, Gemm, ElementwiseOp
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.language.utils import gemm_scale_bytes
from rooflang.programs.presets.b300 import B300Cluster
from rooflang.runtime.simulator import Simulator, OOMError


def build(*, overlap=False, remote=True, ordered=False, cap_bytes=None):
    hw = B300Cluster(1)
    gpus = sorted((c for c in hw.nodes if getattr(c, 'kind', '') == 'gpu'), key=lambda c: c.name)
    if cap_bytes is not None:
        hw.find_local_memory(gpus[0]).capacity_gb = cap_bytes / 1e9
    m, d, k, experts, topk = 64, 4096, 2048, 256, 6
    combine = TokenCombine(m, d, experts, topk)
    combine.inputs = {f'i{i}': Tensor('bf16', (Fraction(m * topk, experts), d)) for i in range(experts)}
    combine.outputs = {'y': Tensor('bf16', (m, d))}
    shared = Gemm(m, d, k, 'fp8', 'bf16')
    shared.inputs = {'x': Tensor('bf16', (m, k))}
    shared.weights = {'w': Tensor('fp8', (k, d)), 's': Tensor('ue8m0', (int(gemm_scale_bytes(d, k, 'fp8')),))}
    shared.outputs = {'y': Tensor('bf16', (m, d))}
    add = ElementwiseOp(m, d, 'bf16')
    add.inputs = {'a': Tensor('bf16', (m, d)), 'b': Tensor('bf16', (m, d))}
    add.outputs = {'y': Tensor('bf16', (m, d))}
    graph = ComputeGraph()
    nodes = {'routed_combine': combine, 'shared_down': shared, 'moe_add': add}
    for kernel in nodes.values(): graph.add_kernel(kernel)
    graph.add_data_edge(combine, add, {'y': 'a'})
    graph.add_data_edge(shared, add, {'y': 'b'})
    if ordered: graph.add_control_edge(combine, shared)
    p = Placement(hardware=hw, graph=graph)
    p.set_kernel_device(combine, gpus[0], stream=0)
    p.set_kernel_device(shared, gpus[0], stream=1 if overlap else 0)
    p.set_kernel_device(add, gpus[0], stream=0)
    for i, t in enumerate(combine.inputs.values()):
        owner = i // 32 if remote else 0
        p.set_tensor_memory(t, hw.find_local_memory(gpus[owner]))
    graph.validate()
    p.validate(graph)
    return graph, p, hw, nodes


def contract(graph, p, hw, nodes):
    """All modeled state frozen except stream ids (the allowed action)."""
    names = {k:n for n,k in nodes.items()}
    node_rows = []
    for name, k in nodes.items():
        assignment = p.get_kernel_device(k)
        row = dict(name=name, type=type(k).__name__, flops=k.flops,
                   weight_read_fraction=str(k.weight_read_fraction),
                   device=assignment.device.name, cap=assignment.resource_cap,
                   attributes={key:str(value) for key,value in vars(k).items() if key not in ('inputs','weights','outputs')}, tensors={})
        for group in ('inputs','weights','outputs'):
            row['tensors'][group] = {
                port:dict(dtype=t.dtype,shape=[str(x) for x in t.shape],bytes=float(t.size_bytes),memory=p.get_tensor_memory(t).name,
                          read_fraction=str(k.input_read_fraction(port)) if group=='inputs' else None)
                for port,t in getattr(k,group).items()}
        node_rows.append(row)
    fabrics = []
    for _, _, attrs in hw._graph.edges(data=True):
        for edge in attrs['fabrics']:
            fabrics.append(dict(name=edge.name, src=edge.src.name, dst=edge.dst.name,
                                read_bw=edge.dst_to_src_bandwidth_gbs,
                                write_bw=edge.src_to_dst_bandwidth_gbs,
                                duplex=edge.is_full_duplex, alpha_us=edge.alpha_us))
    data = dict(nodes=node_rows,edges=sorted((names[a],names[b],sorted(attrs['mapping'].items())) for a,b,attrs in graph._dag.edges(data=True)),
                hardware=sorted((c.name,repr(vars(c))) for c in hw.nodes),
                fabrics=sorted(fabrics, key=lambda e:(e['src'],e['dst'],e['name'])))
    raw=json.dumps(data,sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def run_case(label, **kwargs):
    g,p,hw,nodes=build(**kwargs)
    signature=contract(g,p,hw,nodes)
    names={k:n for n,k in nodes.items()}
    try:
        result=Simulator(g,p,hw).run()
    except OOMError as error:
        return dict(label=label,status='oom',signature=signature,error=str(error),used_bytes=error.used_bytes,capacity_bytes=error.capacity_bytes)
    trace=[dict(name=names[e.kernel],device=e.device.name,stream=e.stream,start_us=e.start_us,end_us=e.end_us,
                compute_us=e.compute_time_us,memory_us=e.memory_time_us,network_us=e.network_time_us) for e in result.trace]
    by_name={e['name']:e for e in trace}
    for src,dst in g._dag.edges:
        assert by_name[names[dst]]['start_us']+1e-8 >= by_name[names[src]]['end_us'], 'A dependency was violated'
    links={}
    # Sum scheduled path traffic; concurrency changes rates, not byte counts.
    sim=Simulator(g,p,hw)
    for kernel in nodes.values():
        dev=p.get_kernel_device(kernel).device
        _,_,_,_,link_data=sim._base_times(kernel,dev,[])
        for (edge,direction),(size,bw) in link_data.items():
            key=f'{edge.src.name}->{edge.dst.name}:{direction}'
            links[key]=links.get(key,0)+size
    return dict(label=label,status='ok',signature=signature,total_us=result.total_time_us,
                peak_bytes={memory.name:float(size) for memory,size in result.peak_memory.items()},
                peak_hbm0_bytes=float(next(v for m,v in result.peak_memory.items() if m.name=='n0-hbm3e-0')),
                flops=sum(k.flops for k in nodes.values()),link_byte_hops=sum(links.values()),trace=trace)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=Path('results/autonomous_optimization_mvp'))
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for scenario,kw in [('remote_overlap',dict(remote=True)),('local_contention',dict(remote=False)),('dependency_guard',dict(remote=True,ordered=True))]:
        serial=run_case(scenario+'_serial',overlap=False,**kw)
        parallel=run_case(scenario+'_two_streams',overlap=True,**kw)
        assert serial['status']==parallel['status']=='ok'
        assert serial['signature']==parallel['signature'], 'The candidate changed forbidden state'
        assert serial['flops']==parallel['flops']
        assert serial['link_byte_hops']==parallel['link_byte_hops']
        if scenario=='remote_overlap': assert parallel['total_us'] < serial['total_us']
        else: assert parallel['total_us'] >= serial['total_us']-1e-8
        rows.extend([serial,parallel])
    baseline,candidate=rows[:2]
    if candidate['peak_hbm0_bytes']>baseline['peak_hbm0_bytes']:
        limit=(candidate['peak_hbm0_bytes']+baseline['peak_hbm0_bytes'])/2
        a=run_case('capacity_guard_serial',overlap=False,remote=True,cap_bytes=limit)
        b=run_case('capacity_guard_two_streams',overlap=True,remote=True,cap_bytes=limit)
        assert a['status']=='ok' and b['status']=='oom'
        assert a['signature']==b['signature']
        rows.extend([a,b])
    summary=dict(scope='Isolated DSV4 Flash MoE fragment. All root inputs ready at t=0. Not a full-model benchmark or numerical-equivalence test.',
                 candidate_action='shared_down stream: 0 -> 1; all operators, dependencies, tensors and device/memory placement fixed',
                 latency_reduction_pct=100*(1-candidate['total_us']/baseline['total_us']),
                 peak_hbm0_delta_bytes=candidate['peak_hbm0_bytes']-baseline['peak_hbm0_bytes'],
                 rows=rows,checks='PASS')
    (args.output_dir/'results.json').write_text(json.dumps(summary,indent=2))
    with (args.output_dir/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['label','status','total_us','peak_hbm0_bytes','flops','link_byte_hops'],lineterminator='\n')
        writer.writeheader();writer.writerows({k:r.get(k,'') for k in writer.fieldnames} for r in rows)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
    for r in rows: print(r['label'],r['status'],r.get('total_us'),r.get('peak_hbm0_bytes'))

if __name__=='__main__': main()
