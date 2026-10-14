"""Three independent, horizontal construction flows for DSV4 Flash decode."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
OUT=Path(__file__).resolve().parent
mx=ET.Element('mxfile',host='app.diagrams.net',version='24.7.17',type='device')
COL={'blue':('#E5F2FF','#B8CFE2'),'purple':('#EDE9FE','#7C3AED'),'green':('#DCFCE7','#16A34A'),'orange':('#FEF3C7','#D97706'),'gray':('#F1F5F9','#64748B'),'panel':('#F8FAFC','#CBD5E1'),'text':('white','white')}
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


# These are construction steps, not a runtime timeline.
import copy

W, H = 1860, 690

def page(name, heading, subtitle, labels):
    p = Page(name, W, H)
    title(p, heading, subtitle)
    for i, label in enumerate(labels):
        p.b(f's{i}', label, 30 + i * 260, 250, 220, 115, size=20)
        if i:
            p.e(f's{i-1}', f's{i}', 'R', 'L')
    return p

def note(p, i, heading, detail):
    x = 30 + i * 610
    p.b(f'note-panel{i}', '', x, 435, 590, 170, 'panel')
    p.b(f'note-title{i}', heading, x+10, 445, 570, 40, 'text', 22)
    p.b(f'note-body{i}', detail, x+10, 492, 570, 110, 'text', 19)

def save(p, basename):
    p.render(basename + '.png')
    document = ET.Element('mxfile', host='app.diagrams.net', version='24.7.17', type='device')
    diagram = next(d for d in mx.findall('diagram') if d.get('name') == p.name)
    document.append(copy.deepcopy(diagram))
    ET.indent(document)
    dest = OUT / (basename + '.drawio')
    ET.ElementTree(document).write(dest, encoding='utf-8', xml_declaration=True)
    root = ET.parse(dest).getroot()
    assert len(root.findall('diagram')) == 1
    cells = root.findall('.//mxCell')
    ids = [c.get('id') for c in cells]
    assert len(ids) == len(set(ids))
    for c in cells:
        if c.get('edge') == '1':
            assert c.get('source') in ids and c.get('target') in ids
    print(dest.name)

p = page('Decode计算图构建', '① Decode：原始计算图如何构建',
    'declare_model(decode=True)  ·  示例：batch=8，历史 KV 长度=8192，本次每条序列处理 1 个 token', [
    '初始化计算图 g\n设置 B=8、S=1\n历史长度=8192',
    '构建 token 入口\nReadInput\n→ Embedding',
    '构建 43 层\nAttention + MoE\n连接残差与层间边',
    '接入历史 KV\n构建缓存读取算子\n连接每层 Attention',
    '构建输出头\nfinal_norm → logits\n→ sampling',
    '构建 KV 保活屏障\n连接 KV 与采样结果\n约束缓存生命周期',
    '校验并返回\n原始计算图 g\n及后续优化所需引用',
])
p.b('config', '输入：模型配置 + Decode 参数', 30, 160, 480, 55, 'gray', 20)
p.e('config', 's0', 'B', 'T', [(270,230),(140,230)])
note(p, 0, '图里包含什么？', '算子节点 + 张量数据边 + 控制依赖。\n先描述一次 Decode 需要做哪些计算；\n此时还没有将算子分配到具体 GPU。')
note(p, 1, '为什么读历史 KV？', 'Attention 使用当前 Q 和历史 KV。\n当前 Decode 路径把历史 KV 作为只读输入；\n未建模追加当前 token 的 KV 更新过程。')
note(p, 2, '返回值不只有 g', 'g, layers, emb, read_input,\nkv_cache_reads, output_head\n这些引用方便优化器定位并修改对应算子。')
p.b('foot', '箭头表示 Python 构图步骤。KV 读取在构图时后接入，不代表运行时要等 43 层算完才读取。', 30, 625, 1800, 45, 'text', 19)
save(p, 'DSV4-Flash-decode-build-flow')

p = page('Decode计算图优化', '② Decode：原始图如何变成并行计算图',
    'optimize_model_cluster_decode(...) 的图变换部分  ·  DP=4，CP=2，EP=8，pp_partition=[43]，GPU=8', [
    '接收原始图 g\n及各类算子引用\n来自 declare_model',
    '校验并行参数\n估算专家权重读取\n组织阶段与 GPU 组',
    '先按 DP 切分\nB=8 → 4 组 B=2\n整理切分产生的通信',
    '各 DP 组内做 CP\nKV 按上下文切分\n复制 Q 路径的计算',
    '切分后续普通算子\n组内 batch 一分为二\n每份 B=1',
    '整理通信与依赖\n融合 Reduce + Scatter\nKV 预载 → token 读取',
    '得到变换后的 g\n含并行副本与通信\n继续建立 Placement',
])
p.b('hw', '硬件图 hw + DP / CP / EP / PP 参数', 290, 160, 600, 55, 'gray', 20)
p.e('hw','s1','B','T',[(590,230),(400,230)])
note(p, 0, 'DP：分不同请求', '8 条序列分成 4 组，每组 2 条。\nEmbedding、层内算子、输出头和 KV 输入\n按相应规则变成多个子图。')
note(p, 1, 'CP：同组协作处理 Attention', '两份 Attention 各读取半份上下文 KV。\nQ 路径复制；后续普通算子按 batch 切分。\nAttention 输出可形成 ReduceScatter。')
note(p, 2, 'EP 与 PP 在哪里体现？', 'EP=8：后续 Placement 将专家分给 8 卡。\nPP=[43]：一个阶段包含全部 43 层；\n不是 43 个阶段，也不再乘出 8 倍 GPU。')
p.b('foot', '这是对同一个 g 的原地变换。“优化后 g”是概念上的中间状态；函数完成 Placement 后才返回 (g, p)。', 30, 625, 1800, 45, 'text', 19)
save(p, 'DSV4-Flash-decode-optimize-flow')

p = page('Decode Placement构建', '③ Placement：并行图如何映射到硬件',
    '同一优化函数的放置部分  ·  一个 H200 节点、GPU0–7、43 层都属于 stage 0', [
    '创建 Placement\np = Placement(...)\n绑定 hw 与 g',
    '放置输入与缓存\nKV 读算子 → 各 GPU\n标记 SSD / DRAM 来源',
    '放置普通算子\nDP：GPU 0/2/4/6\nDP×CP：GPU 0–7',
    '放置专家与路由\n256 个专家分到 8 卡\n每卡每层 32 个专家',
    '处理流水阶段边界\n与 KV 保活张量位置\n本例没有跨阶段边界',
    '补齐通信张量位置\n校验 g 与 p\n记录 KV 显存占用',
    '返回 (g, p)\n连同硬件图 hw\n交给 Simulator',
])
p.b('hw', '硬件图 hw：GPU、HBM、DRAM、SSD 与链路', 30, 160, 700, 55, 'gray', 20)
p.e('hw','s0','B','T',[(380,230),(140,230)])
note(p, 0, '算子 → GPU', '每个 DP 组的代表 GPU 是 dp_rank × 2。\nEmbedding、输出头和 DP-only 算子在代表卡；\nQ / Attention 等副本分布在全部 8 张卡。')
note(p, 1, '张量 → 内存', 'token 输入来源为 DRAM，历史 KV 来源为 SSD。\n预加载后，KV 分片位于各卡本地 HBM；\n通信端点位置决定需要走哪些硬件链路。')
note(p, 2, '此时还没有执行模型', 'Placement 保存设备和内存映射。\nSimulator 随后结合 g、p、hw 模拟计算与传输；\n箭头顺序不是各 GPU 的实际运行时间线。')
p.b('foot', '三张图串起来：declare_model → 原始 g → 并行变换后的 g → Placement → (g, p, hw) → Simulator。', 30, 625, 1800, 45, 'text', 19)
save(p, 'DSV4-Flash-decode-placement-flow')
