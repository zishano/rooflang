"""Generate an editable draw.io diagram and PNG of the H200SuperChip hardware graph."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
W,H = 1900,1440
mx=ET.Element('mxfile',host='app.diagrams.net',version='24.7.17',type='device')
d=ET.SubElement(mx,'diagram',id='h200-superchip',name='H200SuperChip 聚合硬件图')
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
    c=ET.SubElement(r,'mxCell',id=id,value='',edge='1',parent='1',source=src,target=dst,style=f'edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=none;endFill=0;strokeWidth=2;strokeColor={color};exitX={sx};exitY={sy};entryX={tx};entryY={ty};'+('dashed=1;' if dashed else ''))
    g=ET.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
    if via:
        arr=ET.SubElement(g,'Array',attrib={'as':'points'})
        for x,y in via:ET.SubElement(arr,'mxPoint',x=str(x),y=str(y))
    links.append((src,dst,ss,ds,list(via),dashed))

def chain(ids):
    for a,b in zip(ids,ids[1:]):edge(a,b)


box('title','H200SuperChip：单个八 GPU 节点的聚合硬件图',35,20,1830,60,'text',30)
box('subtitle','按 programs/presets/h200.py 绘制：9 个硬件组件、8 条双向链路。名称中的 SuperChip 表示代码聚合抽象。',35,90,1830,55,'text',20)
box('node-panel','',40,170,1400,740,'panel')
box('node-title','一个 H200 节点的聚合表示',65,185,1200,50,'text',25)
box('hbm','聚合 HBM\n8 份 HBM',820,270,245,90,'green',22)
box('gpu','聚合 GPU\n8 张 H200',820,430,245,90,'blue',22)
box('hgx','HGX PCIe Switch',820,605,245,85,'orange',21)
box('cpu-switch','CPU PCIe Switch',520,605,245,85,'orange',21)
box('cpu','聚合 CPU\n2 颗 CPU',200,605,245,85,'purple',22)
box('dram','聚合 DDR5\n2 份 DDR5',200,785,245,85,'green',22)
box('ssd','聚合 SSD\n8 块 SSD',820,785,245,85,'gray',22)
box('nic','聚合 NIC\n8 张 ConnectX-7',1140,605,245,85,'blue',21)
box('ib','InfiniBand Switch',1540,605,310,85,'orange',22)
edge('gpu','hbm','T','B')
edge('gpu','hgx')
edge('cpu','cpu-switch','R','L')
edge('cpu-switch','hgx','R','L')
edge('cpu','dram')
edge('ssd','hgx','T','B')
edge('nic','hgx','L','R')
edge('nic','ib','R','L')
box('link-label','InfiniBand',1400,565,175,40,'text',17)
box('external','代码中的网络交换机\n不代表另一个已建模节点',1480,740,385,90,'text',19)
box('note','八张 GPU 已合并为一个计算节点，因此不再显式表示 NVSwitch；两颗 CPU 已聚合，因此没有 CPU 间 QPI。',40,945,1820,60,'text',21)
box('spec','聚合资源规格（全部来自代码）\nGPU：FP4 / FP8 15832 TFLOPS；\nBF16 / FP16 7916 TFLOPS；FP32 3956 TFLOPS。\nCPU：BF16 / FP16 510.58 TFLOPS；INT8 1022.36 TOPS。\nHBM：1152 GB；DDR5：3072 GB；SSD：2048000 GB。\n内部 GPU 差异、NVSwitch 竞争及 CPU NUMA 差异被隐藏。',40,1050,865,280,'text',20)
box('links-note','统一链路参数（带宽 GB/s；α 为 μs）\nGPU ↔ HBM：38400 / 方向；α 0.5；非全双工。\n4 条普通 PCIe：512 / 方向；α 0.5；全双工。\nNIC ↔ IB Switch：400 / 方向；α 1；全双工。\nCPU ↔ DDR5：665.6 / 方向；α 0.1；非全双工。\nSSD → HGX Switch：112；反向：56；α 50；全双工。\n普通 PCIe：GPU–HGX、NIC–HGX、CPU Switch–HGX、CPU–CPU Switch。',950,1050,910,300,'text',20)
box('footer','图中连线表示可双向传输。非全双工链路的双向流量共享资源；聚合模型不意味着所有资源在实机上可无损合并。',40,1360,1820,60,'text',18)
# Check references and serialize.
allcells=r.findall('mxCell'); ids=[c.get('id') for c in allcells];assert len(ids)==len(set(ids))
for src,dst,*_ in links:assert src in vertices and dst in vertices
ET.indent(mx);ET.ElementTree(mx).write(OUT/'H200SuperChip-topology.drawio',encoding='utf-8',xml_declaration=True)
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
for id,rec in vertices.items():
    if rec[5]!='panel':render_box(rec)
im.save(OUT/'H200SuperChip-topology.png')
print(f'Generated draw.io and PNG: {len(vertices)} elements, {len(links)} edges, {im.size}')
