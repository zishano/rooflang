import xml.etree.ElementTree as ET
from pathlib import Path
mx=ET.Element('mxfile',host='app.diagrams.net',version='24.7.17',type='device')
d=ET.SubElement(mx,'diagram',id='h200-full',name='H200Cluster 完整拓扑')
m=ET.SubElement(d,'mxGraphModel',dx='2400',dy='1500',grid='1',gridSize='10',guides='1',tooltips='1',connect='1',arrows='1',fold='1',page='1',pageScale='1',pageWidth='2400',pageHeight='1650',math='0',shadow='0')
r=ET.SubElement(m,'root'); ET.SubElement(r,'mxCell',id='0'); ET.SubElement(r,'mxCell',id='1',parent='0')
colors={'gpu':('#DBEAFE','#2563EB'),'memory':('#DCFCE7','#16A34A'),'switch':('#FEF3C7','#D97706'),'cpu':('#EDE9FE','#7C3AED'),'nic':('#CFFAFE','#0891B2'),'ssd':('#F1F5F9','#64748B')}
def box(id,text,x,y,w,h,kind='switch',style=''):
    fill,stroke=colors.get(kind,('#FFFFFF','#CBD5E1'))
    c=ET.SubElement(r,'mxCell',id=id,value=text,vertex='1',parent='1',style=f'rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontSize=20;spacing=8;'+style)
    ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
def text(id,value,x,y,w,h,size=16):
    box(id,value,x,y,w,h,'text',f'strokeColor=none;fillColor=none;fontSize={size};align=left;verticalAlign=top;')
def edge(id,src,dst,label,color,points=(),style=''):
    c=ET.SubElement(r,'mxCell',id=id,value='',tooltip=label,edge='1',parent='1',source=src,target=dst,style=f'edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;startArrow=none;endArrow=none;strokeColor={color};strokeWidth=2;fontSize=12;labelBackgroundColor=#FFFFFF;'+style)
    g=ET.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
    if points:
        a=ET.SubElement(g,'Array',attrib={'as':'points'})
        for x,y in points: ET.SubElement(a,'mxPoint',x=str(x),y=str(y))
text('title','<b>H200Cluster：集群与节点内互联</b>',40,20,2300,50,28)
text('sub','示意 n_nodes ≥ 2：每个节点有 8 张 GPU，仅展开第 0、7 路；省略号表示独立且结构相同的设备。',40,80,2300,45,20)
for node,x in [(0,40),(1,1260)]:
    pre=f'n{node}-'
    box(pre+'frame','',x,150,1050,840,'text','fillColor=#F8FAFC;strokeColor=#CBD5E1;dashed=1;')
    text(pre+'title',f'<b>节点 {node}：8 张 H200 GPU</b>',x+20,165,850,40,24)
    box(pre+'nv','NVSwitch（节点内共享）',x+300,225,450,65)
    for i,lx in [(0,x+220),(7,x+620)]:
        for name,caption,y,kind in [('g',f'GPU {i}',340,'gpu'),('h',f'HBM {i}',445,'memory'),('p',f'PCIe Switch {i}',565,'switch'),('s',f'SSD {i}',680,'ssd'),('nic',f'NIC {i}',810,'nic')]:
            box(pre+name+str(i),caption,lx,y,180,60,kind)
        edge(pre+'nv'+str(i),pre+'g'+str(i),pre+'nv','NVLink','#2563EB',[(lx+90,310),(x+525,310)],'exitX=0.5;exitY=0;entryX=0.5;entryY=1;')
        edge(pre+'hbm'+str(i),pre+'g'+str(i),pre+'h'+str(i),'HBM','#16A34A',style='exitX=0.5;exitY=1;entryX=0.5;entryY=0;')
        edge(pre+'gp'+str(i),pre+'g'+str(i),pre+'p'+str(i),'PCIe','#64748B',[(lx+195,370),(lx+195,595)],'exitX=1;exitY=0.5;entryX=1;entryY=0.5;')
        edge(pre+'ssd'+str(i),pre+'p'+str(i),pre+'s'+str(i),'PCIe','#64748B',style='exitX=0.5;exitY=1;entryX=0.5;entryY=0;')
        edge(pre+'niclink'+str(i),pre+'p'+str(i),pre+'nic'+str(i),'PCIe','#0891B2',[(lx+15,640),(lx+15,780)],'exitX=0.08;exitY=1;entryX=0.08;entryY=0;')
        edge(pre+'ib'+str(i),pre+'nic'+str(i),'ib','InfiniBand','#0891B2',[(lx+90,1020),(1200,1020)],'exitX=0.5;exitY=1;entryX=0.5;entryY=0;')
    for cpu,cx,i in [(0,x+20,0),(1,x+850,7)]:
        box(pre+'cpu'+str(cpu),f'CPU {cpu}',cx,565,170,60,'cpu')
        box(pre+'dram'+str(cpu),f'DDR5 {cpu}',cx,715,170,60,'memory')
        edge(pre+'dramlink'+str(cpu),pre+'cpu'+str(cpu),pre+'dram'+str(cpu),'DRAM','#7C3AED',style='exitX=0.5;exitY=1;entryX=0.5;entryY=0;')
        edge(pre+'cpulink'+str(cpu),pre+'cpu'+str(cpu),pre+'p'+str(i),'PCIe','#7C3AED',style=('exitX=1;exitY=0.5;entryX=0;entryY=0.5;' if cpu==0 else 'exitX=0;exitY=0.5;entryX=1;entryY=0.5;'))
    edge(pre+'qpi',pre+'cpu0',pre+'cpu1','QPI','#7C3AED',[(x+10,595),(x+10,930),(x+1040,930),(x+1040,595)],'exitX=0;exitY=0.5;entryX=1;entryY=0.5;dashed=1;')
    for y in [350,455,575,690,820]:
        text(pre+'dots'+str(y),'⋯',x+460,y,120,50,34)
    text(pre+'omitted','中间省略第 1–6 路，每路各有 GPU / HBM / PCIe Switch / SSD / NIC。',x+160,880,740,40,16)
text('more-nodes','⋯',1125,510,100,65,48)
text('more-label','更多节点',1105,595,145,45,16)
box('ib','全局共享 InfiniBand Switch',850,1080,700,80)
text('scope','蓝色 NVSwitch 链路：只连接本节点 GPU。青色 IB 链路：经各自 NIC 连接全局 IB Switch，实现跨节点通信。',40,1190,2300,45,20)
text('mapping','CPU 0 实际连接本节点 PCIe Switch 0–3；CPU 1 实际连接 4–7。图中仅显示到第 0、7 路的连接，其余随设备一同省略。',40,1245,2300,45,18)
text('legend','<b>统一参数说明（GB/s；α 为 μs）</b><br>GPU–HBM：4800，α 0.5，非全双工；GPU–NVSwitch：450，α 0.5，全双工。<br>GPU/NIC/CPU–PCIe Switch：64，α 0.5，全双工；NIC–IB Switch：50，α 1，全双工。<br>SSD→Switch：14，反向 7，α 50，全双工；CPU–DDR5：332.8，α 0.1，非全双工；CPU–CPU：192，α 0.05，全双工。',40,1310,2300,155,18)
text('specs','<b>设备规格（来自代码预设）</b><br>每 GPU：H200 SXM；FP4/FP8 1979、BF16/FP16 989.5、FP32 494.5 TFLOPS。每 CPU：Xeon 6767P；BF16/FP16 255.29、INT8 511.18。<br>每 HBM 144 GB；每 DDR5 1536 GB；每 SSD 256000 GB；每 NIC 为 ConnectX-7（400 Gb/s）。默认 n_nodes=1 时只有一个节点。',40,1480,2300,120,18)
path=Path(__file__).resolve().parent / 'H200Cluster-full-topology.drawio'
ET.indent(mx); ET.ElementTree(mx).write(path,encoding='utf-8',xml_declaration=True)
cells=ET.parse(path).findall('.//mxCell'); ids={c.get('id') for c in cells}
assert len(ids)==len(cells), 'Duplicate cell IDs'
edges=[c for c in cells if c.get('edge')=='1']
for c in edges: assert c.get('source') in ids and c.get('target') in ids
assert len(edges)==34
print(f'Validated: {len(edges)} visible links; omitted devices and links explicitly explained')
