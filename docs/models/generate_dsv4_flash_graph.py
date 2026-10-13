"""Generate an editable draw.io diagram and PNG of the modeled decode graph."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
W,H = 2800,1740
mx=ET.Element('mxfile',host='app.diagrams.net',version='24.7.17',type='device')
d=ET.SubElement(mx,'diagram',id='dsv4-flash-decode',name='DSV4 Flash 计算图分层展开')
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

box('title','DSV4 Flash：从完整模型到 Attention / MoE 子图',30,20,2740,55,'text',32)
box('subtitle','以仓库 dsv4_flash 的单步 Decode 为主；B 为 batch size，S=1。实线：数据依赖；灰色虚线：展开关系。',30,85,2740,50,'text',22)
for id,x,w,title in [('whole',30,350,'① 整体模型'),('layer',430,400,'② 第 i 层'),('attention',880,780,'③ Attention 展开'),('moe',1710,1060,'④ MoE 展开')]:
    box(id+'-panel','',x,160,w,1240,'panel')
    box(id+'-heading',title,x+15,175,w-30,50,'text',26)

# Whole model: one decode step.
for id,label,y,kind in [('read','ReadInput\ntoken ID：B × 1 × 1',255,'green'),('emb','Embedding\nB × 1 × 4096',385,'blue'),('l0','第 0 层',515,'purple'),('ldots','⋯',620,'gray'),('li','中间层 i（其余省略）',715,'purple'),('l42','第 42 层',845,'purple'),('fn','Final RMSNorm',975,'blue'),('logits','Logits 投影\n4096 → 129280',1105,'blue'),('sample','Sampling\n下一个 token ID',1235,'green')]:
    box(id,label,65,y,280,80,kind)
chain(['read','emb','l0','ldots','li','l42','fn','logits','sample'])
# Layer skeleton.
for id,label,y,kind in [('x','本层输入 x\nB × 1 × 4096',255,'green'),('an','attn_norm',385,'blue'),('att','Attention',515,'orange'),('aa','attn_add\nx + Attention 输出',675,'purple'),('ffn','ffn_norm',835,'blue'),('moe-call','MoE',965,'orange'),('fa','ffn_add\n残差 + MoE 输出',1125,'purple'),('out','送往下一层\nB × 1 × 4096',1255,'green')]:
    box(id,label,500,y,260,75,kind)
chain(['x','an','att','aa','ffn','moe-call','fa','out'])
edge('x','aa','L','L',[(460,292.5),(460,712.5)])
edge('aa','fa','R','R',[(800,712.5),(800,1162.5)])
edge('li','x','R','L',[(405,755),(405,292.5)],True)
# Attention Q and new KV paths.
box('a-in','attn_norm 输出',930,255,230,65)
box('qa','wq_a\n4096 → 1024',930,370,230,80)
box('qn','q_norm',930,500,230,65)
box('qb','wq_b\n1024 → 32768',930,615,230,80)
chain(['a-in','qa','qn','qb'])
box('kv-new','wkv\n4096 → 512',1290,370,280,80)
box('kn','kv_norm',1290,500,280,65)
box('sink','kv_sink（Nop）\nDecode 省略追加/更新路径',1255,615,350,80,'gray',18)
edge('a-in','kv-new','R','T',[(1430,287.5)])
chain(['kv-new','kn','sink'])
box('cache','历史 KV cache 读取\nReadInput → Spawn',1270,760,310,80,'green',20)
box('idx','索引 cache 读取\n仅 ratio=4 的层',1270,895,310,80,'green',20)
box('sa','sa：稀疏 Attention\nDpskV4SparseAttn',920,870,300,90,'orange',20)
edge('qb','sa',via=[(1045,820),(1070,820)])
edge('cache','sa','L','R',[(1245,800),(1245,915)])
edge('idx','sa','L','R',[(1245,935),(1245,915)])
box('oa','wo_a：分组输出投影\n32768 → 8192',920,1040,300,85)
box('ob','wo_b\n8192 → 4096',920,1175,300,85)
chain(['sa','oa','ob'])
box('a-note','wo_b 输出送回 attn_add。\nQ/KV 分支由 attn_fan（Spawn）表达。',920,1290,700,75,'text',18)
edge('att','a-in','R','L',[(855,552.5),(855,287.5)],True)
# MoE: gate+dispatch then experts and shared expert.
box('m-in','ffn_norm 输出',2015,255,420,65)
box('gate','gate：4096 → 256\n路由分数（FP32）',1770,395,310,85)
box('dispatch','TokenDispatch\n每个 token 选 6 个专家',1770,565,620,85,'orange')
edge('m-in','gate','B','T',[(2225,350),(1925,350)])
edge('gate','dispatch',via=[(1925,520),(2080,520)])
edge('m-in','dispatch','B','T',[(2225,530),(2080,530)])
box('eu0','专家 0：up\n4096 → 2048',1750,755,245,85)
box('eu255','专家 255：up\n4096 → 2048',2125,755,245,85)
box('expdots','⋯',2010,775,100,50,'text',30)
box('ed0','专家 0：down\n2048 → 4096',1750,910,245,85)
box('ed255','专家 255：down\n2048 → 4096',2125,910,245,85)
for u,dn in [('eu0','ed0'),('eu255','ed255')]:
    ux=point(u,'T')[0]
    edge('dispatch',u,via=[(2080,700),(ux,700)])
    edge(u,dn)
box('combine','TokenCombine\n合并路由专家输出',1875,1075,420,85,'orange')
for dn in ['ed0','ed255']:
    xx=point(dn,'B')[0]
    edge(dn,'combine',via=[(xx,1030),(2085,1030)])
box('shared-up','共享专家 sw_up\n4096 → 2048',2460,565,265,85,'purple')
box('shared-down','共享专家 sw_down\n2048 → 4096',2460,910,265,85,'purple')
edge('m-in','shared-up','R','T',[(2592.5,287.5)])
edge('shared-up','shared-down')
box('m-add','moe_add\n路由专家 + 共享专家',2090,1230,420,85,'green')
edge('combine','m-add',via=[(2085,1190),(2300,1190)])
edge('shared-down','m-add',via=[(2592.5,1190),(2300,1190)])
box('m-note','输出送回 ffn_add；图中省略专家 1–254。',1770,1340,950,45,'text',18)
edge('moe-call','m-in','R','T',[(845,1002.5),(845,145),(2225,145)],True)
# Footer: modeling boundary and port example.
box('foot-left','读图与代码对应\n• add_kernel(k)：加入一个算子；add_data_edge(a, b, {"y": "q"})：a.y → b.q。\n• Spawn 是零成本的分支依赖节点，图中以分叉线简化表示；权重端口未逐一展开。\n• 本图为声明阶段的逻辑计算图，尚未进行 GPU 分配、并行拆分及通信插入。',35,1440,1350,230,'text',20)
box('foot-right','当前实现的建模边界\n• 专家平均负载 M_e = B × S × 6 / 256；并非模拟实际 token 的路由倾斜。\n• KV / index cache 通过 persistence barrier 保持存活至输出完成（辅助边省略）。\n• Prefill 时 S=输入长度，构建压缩 KV，并在最终层后取最后一个 token；本图主线为 Decode。\n• 来源：programs/models/dsv4_flash/model.py、config.py。',1420,1440,1350,250,'text',20)
# Check references and serialize.
allcells=r.findall('mxCell'); ids=[c.get('id') for c in allcells];assert len(ids)==len(set(ids))
for src,dst,*_ in links:assert src in vertices and dst in vertices
ET.indent(mx);ET.ElementTree(mx).write(OUT/'DSV4-Flash-compute-graph.drawio',encoding='utf-8',xml_declaration=True)
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
im.save(OUT/'DSV4-Flash-compute-graph.png')
print(f'Generated draw.io and PNG: {len(vertices)} elements, {len(links)} edges, {im.size}')
