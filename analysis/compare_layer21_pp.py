"""Inspect layer 21 graph placement and simulated communication for two layouts."""
import json
from pathlib import Path
import networkx as nx
from rooflang.programs.models import dsv4_flash as m
from rooflang.programs.presets.h200 import H200Cluster
from rooflang.runtime.simulator import Simulator
from rooflang.language.kernels.comm import CommKernel

out=Path(__file__).resolve().parents[1]/'results/layer21_pp_comparison'
out.mkdir(parents=True,exist_ok=True)
results=[]
for name,dp,ep,partition in [('no_pp',4,8,[43]),('pp2',2,4,[22,21])]:
 print('Simulating',name,flush=True)
 g,l,e,r,k,h=m.declare_model(seq_prefill=8192,decode=True,batch_size=8)
 hw=H200Cluster(1)
 g,p=m.optimize_model_cluster_decode(g,l,hw,e,r,k,h,seq_prefill=8192,cp=2,dp=dp,ep=ep,pp_partition=partition,n_gpus=8)
 sim=Simulator(g,p,hw,measurement_start=r)
 fields={}
 for group in l[21]._decode_copies.values():
  for field,copies in group.items():
   for node in copies:fields[node]=field
 starts=l[21]._decode_copies['dp']['bridge']
 ends=l[21]._decode_copies['cp_dp']['ffn_add']
 descendants=set(starts)
 for node in starts:descendants.update(nx.descendants(g._dag,node))
 ancestors=set(ends)
 for node in ends:ancestors.update(nx.ancestors(g._dag,node))
 nodes=(descendants&ancestors)|set(fields)
 boundary={pred for b in l[22]._decode_copies['dp']['bridge'] for pred in g._dag.predecessors(b)}
 nodes.update(boundary)
 def ports(tensors):
  return {port:dict(shape=[float(x) for x in t.shape],bytes=float(t.size_bytes),memory=p.get_tensor_memory(t).name if p.get_tensor_memory(t) else None) for port,t in tensors.items()}
 def detail(node):
  dev,stream,cap,parts=sim._resolve(node)
  data=dict(field=fields.get(node,''),operator=type(node).__name__,device=dev.name,participants=[d.name for d in parts],inputs=ports(node.inputs),outputs=ports(node.outputs),world=getattr(node,'world',None),total_bytes=getattr(node,'total_bytes',None),transferred_bytes=getattr(node,'transferred_bytes',None))
  return data
 boundary_details=[detail(n) for n in boundary]
 res=sim.run()
 origin=min(x.start_us for x in res.trace if x.kernel in starts)
 events=[]
 for x in res.trace:
  if x.device.name!='n0-nvidia-h200-sxm-0' or x.kernel not in nodes:continue
  if x.network_time_us<=0 and x.kernel not in boundary:continue
  d=detail(x.kernel)
  d.update(start_us=x.start_us-origin,end_us=x.end_us-origin,duration_us=x.end_us-x.start_us,network_base_us=x.network_time_us,network_elapsed_us=x.network_elapsed_time_us,compute_us=x.compute_time_us,hbm_us=x.memory_time_us,outgoing_boundary=x.kernel in boundary)
  if isinstance(x.kernel,CommKernel):
   _,bw,alpha=sim._collective_fabric_info(sim._resolve(x.kernel)[3],False)
   d.update(effective_bandwidth_gbs=bw,alpha_us=alpha)
  events.append(d)
 events.sort(key=lambda d:d['start_us'])
 data=dict(name=name,cp=2,dp=dp,ep=ep,pp_partition=partition,batch_size=8,context=8192,layer=21,decode_us=res.measured_time_us,layer_body_us=max(x.end_us for x in res.trace if x.kernel in ends and x.device.name=='n0-nvidia-h200-sxm-0')-origin,boundaries=sorted(boundary_details,key=lambda d:d['device']),gpu0_events=events)
 results.append(data)
 (out/f'{name}.json').write_text(json.dumps(data,indent=2))
 print(name,'decode',res.measured_time_us,'body',data['layer_body_us'],flush=True)
 for x in events:print(x['field'] or x['operator'], 'boundary',x['outgoing_boundary'],'dur',x['duration_us'],'nt',x['network_base_us'],'bytes',x['total_bytes'],'xfer',x['transferred_bytes'],'world',x['world'],'parts',x['participants'],flush=True)
print('Saved',out,flush=True)
