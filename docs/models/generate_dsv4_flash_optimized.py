"""Multi-page teaching diagram of verified DSV4 Flash decode optimization."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
OUT=Path(__file__).resolve().parent
mx=ET.Element('mxfile',host='app.diagrams.net',version='24.7.17',type='device')
COL={'blue':('#DBEAFE','#2563EB'),'purple':('#EDE9FE','#7C3AED'),'green':('#DCFCE7','#16A34A'),'orange':('#FEF3C7','#D97706'),'gray':('#F1F5F9','#64748B'),'panel':('#F8FAFC','#CBD5E1'),'text':('white','white')}
class Page:
 def __init__(self,name,w,h):
  self.name=name;self.w=w;self.h=h;self.nodes={};self.edges=[]
  d=ET.SubElement(mx,'diagram',id=name,name=name)
  m=ET.SubElement(d,'mxGraphModel',grid='1',gridSize='10',page='1',pageScale='1',pageWidth=str(w),pageHeight=str(h))
  self.root=ET.SubElement(m,'root');ET.SubElement(self.root,'mxCell',id='0');ET.SubElement(self.root,'mxCell',id='1',parent='0')
 def b(self,id,label,x,y,w,h,kind='blue',size=21):
  assert id not in self.nodes
  self.nodes[id]=(label,x,y,w,h,kind,size)
  fill,stroke=COL[kind]
  c=ET.SubElement(self.root,'mxCell',id=id,value=label.replace('\n','<br>'),vertex='1',parent='1',style=f'rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontSize={size};spacing=10;'+('align=left;verticalAlign=top;strokeColor=none;' if kind=='text' else ''))
  ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
 def point(self,id,side):
  _,x,y,w,h,_,_=self.nodes[id]
  return {'T':(x+w/2,y),'B':(x+w/2,y+h),'L':(x,y+h/2),'R':(x+w,y+h/2)}[side]
 def e(self,a,b,ss='B',ds='T',via=(),dash=False):
  assert a in self.nodes and b in self.nodes
  self.edges.append((a,b,ss,ds,via,dash))
  anchors={'T':(.5,0),'B':(.5,1),'L':(0,.5),'R':(1,.5)};sx,sy=anchors[ss];tx,ty=anchors[ds]
  c=ET.SubElement(self.root,'mxCell',id=f'edge-{len(self.edges)}',edge='1',parent='1',source=a,target=b,style=f'edgeStyle=orthogonalEdgeStyle;endArrow=block;endFill=1;rounded=0;strokeColor=#475569;strokeWidth=2;exitX={sx};exitY={sy};entryX={tx};entryY={ty};'+('dashed=1;' if dash else ''))
  g=ET.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
  if via:
   ar=ET.SubElement(g,'Array',attrib={'as':'points'})
   for x,y in via:ET.SubElement(ar,'mxPoint',x=str(x),y=str(y))
 def chain(self,*ids):
  for a,b in zip(ids,ids[1:]):self.e(a,b)
 def render(self,filename):
  s=2;im=Image.new('RGB',(self.w*s,self.h*s),'white');d=ImageDraw.Draw(im)
  xy=lambda p:tuple(round(t*s) for t in p)
  def box(rec):
   text,x,y,w,h,kind,size=rec;fill,stroke=COL[kind]
   if kind!='text':d.rounded_rectangle(xy((x,y,x+w,y+h)),radius=16,fill=fill,outline=stroke,width=2)
   if not text:return
   font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',size*s);lines=[]
   for para in text.split('\n'):
    line=''
    for ch in para:
     if d.textlength(line+ch,font=font)>(w-20)*s and line:lines.append(line);line=ch
     else:line+=ch
    lines.append(line)
   lh=size*1.5;yy=y+8 if kind=='text' else y+(h-len(lines)*lh)/2
   for line in lines:
    xx=(x+10)*s if kind=='text' else (x+w/2)*s-d.textlength(line,font=font)/2
    d.text((xx,yy*s),line,font=font,fill='#172033');yy+=lh
  for rec in self.nodes.values():
   if rec[5]=='panel':box(rec)
  for a,b,ss,ds,via,dash in self.edges:
   pts=[self.point(a,ss),*via,self.point(b,ds)]
   for u,v in zip(pts,pts[1:]):
    length=math.dist(u,v)
    if dash and length:
     for k in range(0,math.ceil(length),14):
      f=k/length;g=min(k+8,length)/length
      d.line([xy((u[0]+(v[0]-u[0])*f,u[1]+(v[1]-u[1])*f)),xy((u[0]+(v[0]-u[0])*g,u[1]+(v[1]-u[1])*g))],fill='#64748B',width=3)
    else:d.line([xy(u),xy(v)],fill='#475569',width=3)
   u,v=pts[-2:];ang=math.atan2(v[1]-u[1],v[0]-u[0]);wings=[(v[0]-12*math.cos(ang)+z*5*math.sin(ang),v[1]-12*math.sin(ang)-z*5*math.cos(ang)) for z in [-1,1]]
   d.polygon([xy(v),*(xy(t) for t in wings)],fill='#475569')
  for rec in self.nodes.values():
   if rec[5]!='panel':box(rec)
  im.save(OUT/filename)

def title(p,t,sub):
 p.b('title',t,30,20,p.w-60,60,'text',30)
 p.b('subtitle',sub,30,85,p.w-60,60,'text',20)

p=Page('01-优化后的算子依赖',2700,1900)
title(p,'DSV4 Flash Decode：优化后的算子依赖','B=8，历史长度=8192，DP=4，CP=2，EP=8，PP=[43]。左侧展开第 2 层（ratio=4）的 DP0；其余三个 DP 组结构相同。')
p.b('att-panel','',30,165,1620,1500,'panel');p.b('moe-panel','',1700,165,970,1500,'panel')
p.b('ah','① 单层 Attention 与残差：DP0 = 序列 0、1',50,180,1580,55,'text',25)
p.b('mh','② MoE：EP 跨全部 8 张 GPU',1720,180,930,55,'text',25)
p.b('entry','来自 Embedding / 上一层 Gather\nbridge → attn_norm → attn_fan\nGPU0；batch=2',530,275,560,110)
p.b('bcast','Broadcast（广播较小的归一化隐藏状态）',500,445,620,75,'orange')
p.e('entry','bcast')
p.b('newkv','wkv → kv_norm → kv_sink\nGPU0；batch=2\n新 KV 更新路径未建模',1240,285,350,115,'gray',19)
p.e('entry','newkv','R','L',[(1170,330),(1170,342.5)])
for rank,x in [(0,210),(1,940)]:
 p.b(f'q{rank}',f'wq_a → q_norm → wq_b\nGPU{rank}；两条序列的 Q 都计算\nQ：(2, 1, 32768)',x,580,490,110)
 p.b(f'kv{rank}',f'已预加载至 HBM{rank}\n主 KV：(2, 1088, 512) FP8\nindex KV：(2, 1024, 128) FP4',x,760,490,110,'green',20)
 p.b(f'sa{rank}',f'Attention CP{rank} / GPU{rank}\nQ 完整，KV 仅本地上下文分片\n输出为 batch=2 的部分结果',x,950,490,110)
 qx=x+245;p.e('bcast',f'q{rank}',via=[(810,550),(qx,550)])
 # Q travels beside cache box; cache is a separate dependency.
 p.e(f'q{rank}',f'sa{rank}','L','L',[(x-35,635),(x-35,1005)])
 p.e(f'kv{rank}',f'sa{rank}')
p.b('rs','ReduceScatter（DP0 内两张 GPU）\n归约 Attention 部分结果，再按 batch 分发',500,1140,620,95,'orange')
p.e('sa0','rs',via=[(455,1100),(810,1100)]);p.e('sa1','rs',via=[(1185,1100),(810,1100)])
for rank,x in [(0,210),(1,940)]:
 p.b(f'out{rank}',f'GPU{rank}：wo_a → wo_b → attn_add\n序列 {rank}；batch=1\nattn_add 同时接收原始 x 的对应分片',x,1320,490,115)
 p.b(f'ffn{rank}',f'GPU{rank}：ffn_bridge → ffn_norm\n→ ffn_fan → MoE 分支',x,1490,490,95,'purple',20)
 p.e('rs',f'out{rank}',via=[(810,1270),(x+245,1270)]);p.e(f'out{rank}',f'ffn{rank}')
p.b('res-note','原始 x 的残差支路：bridge → Scatter → 两个 attn_add（跨线省略）。\nQ 复制由逆序 dup(wq_b, q_norm, wq_a) 实现；Attention 后的 Reduce+Scatter 已融合。',60,1690,1570,140,'text',21)
# MoE aggregate with explicit shared and residual paths.
p.b('route','每 GPU 一个 batch=1 分片\ngate + TokenDispatch；每 token Top-6',1750,280,500,100)
p.b('shared','每 GPU 的共享专家\nsw_up → sw_down',2340,280,280,100,'purple',19)
p.b('gather','Gather 通信链\n汇集各分片发往同一专家的数据',1750,475,500,95,'orange')
p.b('experts','256 个路由专家：up → down\nGPU0：0–31；GPU1：32–63；…\nGPU7：224–255；按平均路由量计费',1750,650,500,140)
p.b('scatter','Scatter 通信链\n将专家结果分回各 token 分片',1750,865,500,95,'orange')
p.b('combine','每 GPU：TokenCombine',1750,1040,500,80)
p.b('moeadd','每 GPU：moe_add\n路由专家输出 + 共享专家输出',1880,1230,650,100,'purple')
p.chain('route','gather','experts','scatter','combine');p.e('combine','moeadd',via=[(2000,1160),(2205,1160)])
p.e('shared','moeadd',via=[(2480,1160),(2205,1160)])
p.b('ffnadd','每 GPU：ffn_add\n加回 ffn_bridge 的残差分片',1880,1410,650,90)
p.b('return','每个 DP 组内 Gather\n恢复 batch=2 → 下一层代表 GPU\n最后一层后接输出头',1850,1550,710,100,'green',19)
p.chain('moeadd','ffnadd','return')
p.b('moenote','ffn_fan 同时供给 gate、dispatch、共享专家；ffn_bridge 还直连 ffn_add。\n为清晰省略重复分支线。本页是依赖示意，不表示各 GPU 的精确开始时刻。',1720,1690,920,155,'text',20)
p.render('DSV4-Flash-optimized-operators.png')

p=Page('02-Placement映射',2500,1750)
title(p,'Placement：同一组 GPU 同时承载 DP、CP 和 EP','一个 H200 节点，43 层都在 PP stage 0。rank = dp_rank × 2 + cp_rank；以下为当前代码的实际放置规则。')
for dp in range(4):
 x=30+dp*615
 p.b(f'group{dp}',f'DP{dp}：序列 {2*dp}、{2*dp+1}',x,175,580,65,'orange',23)
 for cp in range(2):
  rank=dp*2+cp;y=280+cp*350
  label=f'GPU {rank} / CP{cp}\nQ 路径 + Attention：处理组内两条序列\nKV：该 DP 组的第 {cp} 个上下文分片\n输出投影 / MoE 普通算子：序列 {rank}\n每层路由专家：{rank*32}–{rank*32+31}'
  if cp==0:label+='\n另有 DP-only、Embedding 与输出头'
  p.b(f'g{rank}',label,x,y,580,280,'blue' if cp==0 else 'purple',21)
 p.b(f'g-note{dp}',f'代表 GPU = {2*dp}\nDP-only：bridge / attn_norm / attn_fan\nwkv / kv_norm / kv_sink',x,985,580,145,'gray',20)
p.b('memtitle','内存放置与通信',30,1190,2440,55,'text',26)
for id,label,x,w in [('ssd','SSD r\n主 KV / index KV 的输入来源',40,500),('hbm','HBM r\n预加载后的 KV、该 GPU 算子数据\n显式位置可以覆盖默认分配',660,700),('barrier','barrier 的 KV 输入\n仍指向对应层 / rank 的本地 HBM\n保持 cache 存活到输出完成',1500,930)]:
 p.b(id,label,x,1290,w,120,'green',21)
p.e('ssd','hbm','R','L');p.e('hbm','barrier','R','L',dash=True)
p.b('memnote','token 输入：DDR5 0 → GPU0 上的 read_input → 四个 DP Embedding（GPU0/2/4/6）。\nset_kernel_device 会为相关张量分配默认本地内存；set_tensor_memory 设置 SSD、DRAM、路由等特殊位置。\n通信 Kernel 通常无需绑定单个 GPU；根据张量端点与硬件拓扑使用链路。本例单节点，跨 GPU 通信走节点内互联。\n图中 r=0…7；GPU r 对应 SSD r。EP=8 和 DP×CP=8 复用这 8 张卡，不相乘为 64 张卡。',35,1490,2420,215,'text',22)
p.render('DSV4-Flash-optimized-placement.png')

p=Page('03-构建与执行顺序',2500,1830)
title(p,'优化与 Placement 的构建过程，以及之后的仿真执行','左侧在 Python 中修改图和设置映射；右侧由 Simulator 按依赖与资源竞争推进。箭头表示先后关系，不是时间比例。')
p.b('left-panel','',30,165,1160,1360,'panel');p.b('right-panel','',1250,165,1220,1360,'panel')
p.b('lh','① optimize_model_cluster_decode',55,185,1110,60,'text',26)
p.b('rh','② simulate(g, placement, hw)',1275,185,1170,60,'text',26)
steps=[('validate','检查参数 / 设置专家权重读取比例\n建立 layer_stages、stage_gpus',280),('dp','DP：batch 8 → 4 组 × 每组 2 条序列\n拆输入、输出头、每层算子、KV 读取和 barrier',430),('cp','CP：每 DP 组拆到 2 个上下文分片\nQ 路径复制；指定后续算子再按 batch 拆分',590),('comm','optimize_comms + 控制依赖\nReduce+Scatter → ReduceScatter；所有 KV 读取 → read_input',770),('place','创建 Placement\n设 KV 来源 SSD、token 来源 DRAM、算子 GPU、专家归属',950),('finish','补齐通信 / barrier 的内存位置\n校验图与 Placement，记录 KV 占用',1130),('result','返回修改后的 g 与 placement',1320)]
for id,label,y in steps:p.b(id,label,100,y,1010,110,'orange' if id in ['cp','comm'] else 'blue',23)
p.chain(*(id for id,_,_ in steps))
run=[('preload','预加载：所有层的主 KV / index KV\nSSD → 对应 rank 的 HBM',280),('read','所有预加载依赖完成 → read_input\n默认 measured Decode 从此开始计时',480),('embed','四个 DP Embedding 在代表 GPU 上执行',680),('layers','第 0 层 → 第 1 层 → … → 第 42 层\n每层按第 1 页的分片与通信依赖推进',860),('head','最后一层 Gather → 每 DP 组的输出头\nFinal RMSNorm → Logits → Sampling',1080),('barrier2','8 个 barrier 分片等待 cache 与输出\n完成后汇总 trace、耗时和峰值内存',1300)]
for id,label,y in run:p.b(id,label,1330,y,1060,120,'green' if id in ['preload','barrier2'] else 'purple',23)
p.chain(*(id for id,_,_ in run))
p.b('audit','结构核验\n实际构建 dsv4_flash：B=8、seq_prefill=8192、DP=4、CP=2、EP=8、PP=[43]。\n声明图 23139 个 Kernel；优化后 141043 个 Kernel（含通信与结构节点）。第 2 层实际后继：sa → ReduceScatter → wo_a。\n已通过 g.validate() / placement.validate()，并核对分片 shape、GPU 映射及专家归属；未运行本配置的耗时仿真。',40,1570,2420,220,'text',22)
p.render('DSV4-Flash-optimized-process.png')
ET.indent(mx)
path=OUT/'DSV4-Flash-optimized-decode.drawio'
ET.ElementTree(mx).write(path,encoding='utf-8',xml_declaration=True)
for root in ET.parse(path).findall('.//root'):
 cells=root.findall('mxCell');ids={c.get('id') for c in cells};assert len(ids)==len(cells)
 for c in cells:
  if c.get('edge')=='1':assert c.get('source') in ids and c.get('target') in ids
print('Generated and validated 3 draw.io pages and 3 PNG previews')
