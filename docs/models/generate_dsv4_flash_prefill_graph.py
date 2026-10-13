"""Generate an editable draw.io diagram and PNG of the modeled prefill graph."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
W,H = 3600,2220
mx=ET.Element('mxfile',host='app.diagrams.net',version='24.7.17',type='device')
d=ET.SubElement(mx,'diagram',id='dsv4-flash-prefill',name='DSV4 Flash Prefill 计算图')
m=ET.SubElement(d,'mxGraphModel',grid='1',gridSize='10',page='1',pageScale='1',pageWidth=str(W),pageHeight=str(H),math='0',shadow='0')
r=ET.SubElement(m,'root');ET.SubElement(r,'mxCell',id='0');ET.SubElement(r,'mxCell',id='1',parent='0')
vertices={};links=[]
palette={'blue':('#DBEAFE','#2563EB'),'purple':('#EDE9FE','#7C3AED'),'green':('#DCFCE7','#16A34A'),'orange':('#FEF3C7','#D97706'),'gray':('#F1F5F9','#64748B'),'panel':('#F8FAFC','#CBD5E1'),'text':('#FFFFFF','#FFFFFF')}
def box(id,label,x,y,w,h,kind='blue',size=20):
    fill,stroke=palette[kind]
    c=ET.SubElement(r,'mxCell',id=id,value=label.replace('\n','<br>'),vertex='1',parent='1',style=f'rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontSize={size};fontFamily=Noto Sans CJK SC;spacing=10;'+('align=left;verticalAlign=top;strokeColor=none;' if kind=='text' else ''))
    ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
    vertices[id]=(label,x,y,w,h,kind,size)
    return id

def point(id,side):
    _,x,y,w,h,_,_=vertices[id]
    return {'T':(x+w/2,y),'B':(x+w/2,y+h),'L':(x,y+h/2),'R':(x+w,y+h/2)}[side]

def edge(src,dst,ss='B',ds='T',via=(),dashed=False):
    id=f'e{len(links)}';color='#94A3B8' if dashed else '#475569'
    sx,sy={'T':(.5,0),'B':(.5,1),'L':(0,.5),'R':(1,.5)}[ss]
    tx,ty={'T':(.5,0),'B':(.5,1),'L':(0,.5),'R':(1,.5)}[ds]
    c=ET.SubElement(r,'mxCell',id=id,value='',edge='1',parent='1',source=src,target=dst,style=f'edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeWidth=2;strokeColor={color};exitX={sx};exitY={sy};entryX={tx};entryY={ty};'+('dashed=1;' if dashed else ''))
    g=ET.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
    if via:
        arr=ET.SubElement(g,'Array',attrib={'as':'points'})
        for x,y in via:ET.SubElement(arr,'mxPoint',x=str(x),y=str(y))
    links.append((src,dst,ss,ds,list(via),dashed))

def chain(ids):
    for a,b in zip(ids,ids[1:]):edge(a,b)


box('title','DSV4 Flash Prefill：完整模型、单层、Attention / KV 构建与 MoE',30,20,3540,60,'text',32)
box('subtitle','主图展开 ratio=4 的 Prefill 层，S=输入序列长度。橙色标出阶段差异；实线为依赖，虚线为展开关系。',30,85,3540,50,'text',22)
for id,x,w,title in [('whole',30,350,'① Prefill 整体'),('layer',430,450,'② 第 i 层'),('attention',930,1530,'③ Attention / KV 构建（ratio=4）'),('moe',2510,1060,'④ MoE：处理 B × S 个 token')]:
    box(id+'-panel','',x,160,w,1750,'panel')
    box(id+'-heading',title,x+15,175,w-30,55,'text',26)
for id,label,y,kind in [('read','ReadInput\nB × S × 1',275,'green'),('emb','Embedding\nB × S × 4096',420,'blue'),('l0','第 0 层',565,'purple'),('dots','⋯',705,'gray'),('li','中间层 i（其余省略）',845,'purple'),('l42','第 42 层',990,'purple'),('last','last_token：Slice\nB × S × D → B × 1 × D',1150,'orange'),('final','Final RMSNorm',1320,'blue'),('logits','Logits 投影\n4096 → 129280',1470,'blue'),('sample','Sampling\n每条序列输出首个新 token',1620,'green')]:
    box(id,label,60,y,290,90,kind,19)
chain(['read','emb','l0','dots','li','l42','last','final','logits','sample'])
box('whole-note','43 层处理全部 S 个 token；\n输出头只处理最后一个位置。',50,1770,310,100,'text',18)
for id,label,y,kind in [('x','本层输入 x\nB × S × 4096',275,'green'),('attention-call','Attention 与 KV 构建\n包含归一化、Q/KV、压缩器',520,'orange'),('aa','attn_add\nx + Attention 输出',805,'purple'),('norm','ffn_norm',1020,'blue'),('moe-call','MoE',1220,'blue'),('fa','ffn_add\n残差 + MoE 输出',1490,'purple'),('out','下一层输入\nB × S × 4096',1690,'green')]:
    box(id,label,495,y,320,95,kind,20)
chain(['x','attention-call','aa','norm','moe-call','fa','out'])
edge('x','aa','L','L',[(460,322.5),(460,852.5)])
edge('aa','fa','R','R',[(850,852.5),(850,1537.5)])
edge('li','x','R','L',[(405,890),(405,322.5)],True)
# Explicit three-way bridge.
box('bridge','bridge = Spawn(world=3)\n原始层输入 x：B × S × 4096',1370,275,520,95,'orange')
box('an','attn_norm → attn_fan\n分出 Q 与新 KV 两路',1160,435,420,85)
box('y1','y：Attention 主路',1135,385,300,42,'text',18)
box('y2','y2：压缩器输入',1960,385,320,42,'text',18)
box('y3','y3：保留原始 x 做残差',2070,275,340,70,'text',18)
edge('bridge','an',via=[(1630,405),(1370,405)])
box('comp','comp：压缩器\nB × S × 4096 → B × (S//4) × 512',1950,575,360,95,'orange',19)
box('cn','comp_norm\n压缩 KV：B × (S//4) × 512',1950,725,360,85,'orange',19)
edge('bridge','comp','R','T',[(2130,322.5)])
edge('comp','cn')
for id,label,x,y,w in [('qa','wq_a\n4096 → 1024',990,575,300),('qn','q_norm',990,725,300),('qb','wq_b\n1024 → 32768',990,865,300),('kv','wkv\n4096 → 512',1430,575,320),('kn','kv_norm',1430,725,320),('win','kv_win_slice\n保留窗口 128 个位置',1430,865,320)]:
    box(id,label,x,y,w,85,'blue',20)
edge('an','qa',via=[(1370,550),(1140,550)])
edge('an','kv',via=[(1370,550),(1590,550)])
chain(['qa','qn','qb']);chain(['kv','kn','win'])
box('cat','kv_concat：Concat\n窗口 KV + 压缩 KV',1650,1010,400,85,'orange')
edge('win','cat',via=[(1590,975),(1850,975)])
edge('cn','cat',via=[(2130,975),(1850,975)])
box('quant','kv_cache_quant：Slice\nBF16 → FP8（代码抽象）',1650,1145,400,85,'orange',20)
box('persist','kv_persist_fan：Spawn(3)\ny → Attention；y2 → barrier\ny3 → index cache',1650,1280,400,110,'orange',19)
chain(['cat','quant','persist'])
box('sa','sa：稀疏 Attention\ncausal=True\n输出 B × S × 32768',990,1320,360,110,'orange',20)
edge('qb','sa',via=[(1140,1250),(1170,1250)])
edge('persist','sa','L','R',[(1500,1335),(1500,1375)])
box('islice','index_cache_slice：Slice\nFP4：B × (S//4) × 128',2090,1420,330,85,'orange',18)
box('ifan','index_cache_fan：Spawn(2)\ny → Attention\ny2 → barrier',2090,1560,330,100,'orange',18)
edge('persist','islice','R','T',[(2255,1335)])
edge('islice','ifan')
edge('ifan','sa','L','R',[(2070,1610),(2070,1450),(1510,1450),(1510,1375)])
box('oa','wo_a：分组投影\n32768 → 8192',990,1480,360,85)
box('ob','wo_b\n8192 → 4096',990,1620,360,85)
box('attn-add','attn_add\n原始 x + wo_b 输出',990,1760,360,85,'purple')
chain(['sa','oa','ob','attn-add'])
edge('bridge','attn-add','R','R',[(2435,322.5),(2435,1870),(1390,1870),(1390,1802.5)])
box('barrier','全图 KV persistence barrier（Nop）\n汇集各层 cache 与 Sampling 输出；\n保持 cache 存活到 Prefill 输出完成',1600,1760,760,100,'green',19)
edge('persist','barrier','B','T',[(1850,1690),(1980,1690)])
edge('ifan','barrier','B','T',[(2255,1690),(1980,1690)])
box('sampling-ref','Sampling 输出也接入 barrier（跨区边省略）',1550,1705,840,40,'text',17)
edge('attention-call','bridge','R','L',[(905,567.5),(905,250),(1340,250),(1340,322.5)],True)
# MoE uses the same structure with more tokens.
box('m-in','ffn_norm 输出\nB × S × 4096',2815,275,420,90)
box('gate','gate：4096 → 256\n路由分数（FP32）',2570,485,310,85)
box('dispatch','TokenDispatch\n每个 token 选择 6 个专家',2570,700,620,90,'orange')
edge('m-in','gate',via=[(3025,420),(2725,420)])
edge('gate','dispatch',via=[(2725,640),(2880,640)])
edge('m-in','dispatch','B','T',[(3025,650),(2880,650)])
for id,label,x,y,kind in [('eu0','专家 0：up\n4096 → 2048',2550,935,'blue'),('eu255','专家 255：up\n4096 → 2048',2925,935,'blue'),('ed0','专家 0：down\n2048 → 4096',2550,1140,'blue'),('ed255','专家 255：down\n2048 → 4096',2925,1140,'blue'),('su','共享专家 sw_up\n4096 → 2048',3260,700,'purple'),('sd','共享专家 sw_down\n2048 → 4096',3260,1140,'purple')]:
    box(id,label,x,y,265,90,kind,20)
box('edots','⋯',2820,955,90,50,'text',32)
for u,dn in [('eu0','ed0'),('eu255','ed255')]:
    ux=point(u,'T')[0];edge('dispatch',u,via=[(2880,860),(ux,860)]);edge(u,dn)
edge('m-in','su','R','T',[(3392.5,320)])
edge('su','sd')
box('combine','TokenCombine\n合并路由专家输出',2685,1390,420,90,'orange')
for dn in ['ed0','ed255']:
    xx=point(dn,'B')[0];edge(dn,'combine',via=[(xx,1320),(2895,1320)])
box('m-add','moe_add\n路由专家输出 + 共享专家输出',2890,1650,420,95,'green')
edge('combine','m-add',via=[(2895,1570),(3100,1570)])
edge('sd','m-add',via=[(3392.5,1570),(3100,1570)])
box('m-note','结构与 Decode 相同；M = B × S。\n每个专家平均分到 M_e = B × S × 6 / 256。\n输出送回 ffn_add；中间专家省略。',2570,1770,940,105,'text',19)
edge('moe-call','m-in','R','T',[(895,1267.5),(895,145),(3025,145)],True)
# Explicit differences and limits.
box('f1','相比 Decode 的变化\n1. S 从 1 变为输入长度；本层处理全部输入 token。\n2. 压缩层 bridge 增加 y2 → comp 分支，残差改走 y3。\n3. 由当前输入构建 KV，代替外部历史 KV 读取；Attention 使用 causal=True。\n4. 最终层后新增 last_token Slice，然后才进入输出头。',35,1950,1190,235,'text',20)
box('f2','不同层的条件\n• ratio=4：按主图构建，含压缩器与 index cache。\n• ratio=128：保留压缩器，省略 index cache 分支；\n  kv_persist_fan 为 Spawn(2)。\n• ratio=0：bridge 为 Spawn(2)，省略 comp / Concat / index；\n  窗口 KV 直接接 kv_cache_quant。前两层属于此类。',1270,1950,1120,235,'text',20)
box('f3','读图边界\n• 按仓库 model.py / config.py 绘制，尚未做并行拆分与设备放置。\n• Spawn 为零成本分支；Slice 的格式转换是代码中的建模抽象。\n• 全图只有一个 persistence barrier，这里展示本层输入关系。\n• 未展开权重端口；MoE 使用平均负载，未模拟实际路由倾斜。',2430,1950,1130,235,'text',20)
# Check references and serialize.
allcells=r.findall('mxCell'); ids=[c.get('id') for c in allcells];assert len(ids)==len(set(ids))
for src,dst,*_ in links:assert src in vertices and dst in vertices
ET.indent(mx);ET.ElementTree(mx).write(OUT/'DSV4-Flash-prefill-compute-graph.drawio',encoding='utf-8',xml_declaration=True)
# Render the same geometry with arrows, panels and dashed expansion links.
scale=2; im=Image.new('RGB',(W*scale,H*scale),'white');draw=ImageDraw.Draw(im)
font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def coords(v):return tuple(round(q*scale) for q in v)
def render_box(rec):
    label,x,y,w,h,kind,size=rec;fill,stroke=palette[kind]
    if kind!='text':draw.rounded_rectangle(coords((x,y,x+w,y+h)),radius=16,fill=fill,outline=stroke,width=2)
    if not label:return
    font=ImageFont.truetype(font_path,size*scale);lines=[]
    for para in label.split('\n'):
        line=''
        for ch in para:
            if draw.textlength(line+ch,font=font)>(w-20)*scale and line:lines.append(line);line=ch
            else:line+=ch
        lines.append(line)
    lh=size*1.5;yy=y+8 if kind=='text' else y+(h-len(lines)*lh)/2
    for line in lines:
        xx=(x+10)*scale if kind=='text' else (x+w/2)*scale-draw.textlength(line,font=font)/2
        draw.text((xx,yy*scale),line,font=font,fill='#172033');yy+=lh
for id,rec in vertices.items():
    if rec[5]=='panel':render_box(rec)
for src,dst,ss,ds,via,dashed in links:
    pts=[point(src,ss)]+via+[point(dst,ds)];color='#94A3B8' if dashed else '#475569'
    for a,b in zip(pts,pts[1:]):
        length=math.dist(a,b)
        if dashed and length:
            for k in range(0,math.ceil(length),14):
                f=k/length;g=min(k+8,length)/length
                draw.line([coords((a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f)),coords((a[0]+(b[0]-a[0])*g,a[1]+(b[1]-a[1])*g))],fill=color,width=3)
        else:draw.line([coords(a),coords(b)],fill=color,width=3)
    a,b=pts[-2:];angle=math.atan2(b[1]-a[1],b[0]-a[0]);tip=b
    wings=[(b[0]-12*math.cos(angle)+sgn*5*math.sin(angle),b[1]-12*math.sin(angle)-sgn*5*math.cos(angle)) for sgn in [-1,1]]
    draw.polygon([coords(tip),coords(wings[0]),coords(wings[1])],fill=color)
for id,rec in vertices.items():
    if rec[5]!='panel':render_box(rec)
im.save(OUT/'DSV4-Flash-prefill-compute-graph.png')
print(f'Generated draw.io and PNG: {len(vertices)} elements, {len(links)} edges, {im.size}')
