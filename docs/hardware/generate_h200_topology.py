import xml.etree.ElementTree as ET
from pathlib import Path
mx=ET.Element('mxfile',host='app.diagrams.net',version='24.7.17',type='device')
d=ET.SubElement(mx,'diagram',id='h200-full',name='H200Cluster 完整拓扑')
m=ET.SubElement(d,'mxGraphModel',dx='2400',dy='1500',grid='1',gridSize='10',guides='1',tooltips='1',connect='1',arrows='1',fold='1',page='1',pageScale='1',pageWidth='2400',pageHeight='1850',math='0',shadow='0')
r=ET.SubElement(m,'root'); ET.SubElement(r,'mxCell',id='0'); ET.SubElement(r,'mxCell',id='1',parent='0')
colors={'gpu':('#DBEAFE','#2563EB'),'memory':('#DCFCE7','#16A34A'),'switch':('#FEF3C7','#D97706'),'cpu':('#EDE9FE','#7C3AED'),'nic':('#CFFAFE','#0891B2'),'ssd':('#F1F5F9','#64748B')}
def box(id,text,x,y,w,h,kind='switch',style=''):
    fill,stroke=colors.get(kind,('#FFFFFF','#CBD5E1'))
    c=ET.SubElement(r,'mxCell',id=id,value=text,vertex='1',parent='1',style=f'rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontSize=14;spacing=8;'+style)
    ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
def text(id,value,x,y,w,h,size=16):
    box(id,value,x,y,w,h,'text',f'strokeColor=none;fillColor=none;fontSize={size};align=left;verticalAlign=top;')
def edge(id,src,dst,label,color,points=(),style=''):
    c=ET.SubElement(r,'mxCell',id=id,value='',tooltip=label,edge='1',parent='1',source=src,target=dst,style=f'edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;startArrow=none;endArrow=none;strokeColor={color};strokeWidth=2;fontSize=12;labelBackgroundColor=#FFFFFF;'+style)
    g=ET.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
    if points:
        a=ET.SubElement(g,'Array',attrib={'as':'points'})
        for x,y in points: ET.SubElement(a,'mxPoint',x=str(x),y=str(y))
text('title','<b>H200Cluster：完整单节点拓扑与跨节点互联</b>',40,20,2300,45,26)
text('sub','严格按所贴代码绘制；默认 n_nodes=1。图中每个设备均独立展开，线表示双向连接，具体双工与时延见下方参数表。',40,75,2300,40)
box('node','',265,130,1870,900,'text','fillColor=#F8FAFC;strokeColor=#CBD5E1;dashed=1;')
text('node-label','<b>节点 n（n = 0 … n_nodes−1）</b>',285,145,700,35)
box('nv','<b>逻辑 NVSwitch</b><br>节点内共享',950,190,470,70)
for i in range(8):
    x=300+i*230
    box(f'g{i}',f'<b>GPU {i}</b>',x,320,200,110,'gpu')
    box(f'h{i}',f'<b>HBM {i}</b>',x,470,200,65,'memory')
    box(f'p{i}',f'<b>PCIe Switch {i}</b>',x,610,200,65)
    box(f's{i}',f'<b>SSD {i}</b>',x,730,200,65,'ssd')
    box(f'n{i}',f'<b>NIC {i}</b>',x,885,200,65,'nic')
    edge(f'nv{i}',f'g{i}','nv','NVLink 450','#2563EB',[(x+100,285),(1185,285)],'exitX=0.5;exitY=0;entryX=0.5;entryY=1;')
    edge(f'hbm{i}',f'g{i}',f'h{i}','HBM 4800','#16A34A')
    edge(f'gp{i}',f'g{i}',f'p{i}','PCIe 64','#64748B',[(x+215,375),(x+215,642)],'exitX=1;exitY=0.5;entryX=1;entryY=0.5;')
    edge(f'sp{i}',f's{i}',f'p{i}','读 14 / 写 7','#64748B')
    edge(f'np{i}',f'n{i}',f'p{i}','PCIe 64','#0891B2',[(x+15,850),(x+15,690)],'exitX=0.08;exitY=0;entryX=0.08;entryY=1;')
    edge(f'ib{i}',f'n{i}','ib','IB 50','#0891B2',[(x+100,990),(1185,990)],'exitX=0.5;exitY=1;entryX=0.5;entryY=0;')
box('ib','<b>全局共享 InfiniBand Switch</b><br>跨节点互联',900,1060,570,75)
for s,x in [(0,30),(1,2160)]:
    box(f'c{s}',f'<b>CPU {s}</b>',x,570,210,105,'cpu')
    box(f'd{s}',f'<b>DDR5 {s}</b>',x,755,210,70,'memory')
    edge(f'dram{s}',f'c{s}',f'd{s}','DRAM 332.8','#7C3AED')
    for i in range(s*4,(s+1)*4):
        px=300+i*230
        bus_y=550+(i%4)*10
        edge(f'cp{s}-{i}',f'c{s}',f'p{i}','PCIe 64','#7C3AED',[(135 if s==0 else 2265,bus_y),(px+100,bus_y)],'exitX=0.5;exitY=0;entryX=0.5;entryY=0;')
edge('qpi','c0','c1','QPI：192 GB/s / 方向；α = 0.05 μs；全双工','#7C3AED',[(10,622),(10,1180),(2390,1180),(2390,622)],'exitX=0;exitY=0.5;entryX=1;entryY=0.5;dashed=1;')
text('cpu-map','<b>CPU 归属：</b>CPU 0 → PCIe Switch 0–3；CPU 1 → PCIe Switch 4–7。每个 PCIe Switch 只连接同编号 GPU、NIC、SSD 及所属 CPU。',40,1210,2300,45)
text('table','<b>链路参数（全部带宽单位：GB/s；α 为固定时延）</b><br><br>GPU ↔ HBM：4800 / 方向；α 0.5 μs；非全双工（双向共享资源）<br>GPU ↔ NVSwitch：450 / 方向；α 0.5 μs；全双工<br>GPU ↔ PCIe Switch、NIC ↔ PCIe Switch、CPU ↔ PCIe Switch：64 / 方向；α 0.5 μs；全双工<br>NIC ↔ IB Switch：50 / 方向（400 Gb/s）；α 1.0 μs；全双工<br>SSD → PCIe Switch：14；PCIe Switch → SSD：7；α 50 μs；全双工<br>CPU ↔ DDR5：332.8 / 方向；α 0.1 μs；非全双工<br>CPU 0 ↔ CPU 1：192 / 方向；α 0.05 μs；全双工',40,1270,1550,285)
text('notes','<b>建模说明</b><br><br>• 每节点：8 GPU、8 HBM、8 PCIe Switch、8 NIC、8 SSD、2 CPU、2 DDR5、1 逻辑 NVSwitch。<br>• 全集群：仅一个共享 IB Switch，位于节点循环之外。多节点时复制虚线框及其 CPU / DDR5，每张 NIC 仍接该 IB Switch。<br>• find_aggregate_bandwidth：少于 2 设备返回 ∞；同节点返回 450；跨节点返回 min(各节点参与设备数) × 50 GB/s。<br>• 数字来自代码预设，不代表对真实硬件规格的独立核验。',1620,1270,740,290)
text('specs','<b>设备规格（同类设备统一列出，数值来自代码）</b><br><br>GPU：H200 SXM；FP4 / FP8 1979 TFLOPS；BF16 / FP16 989.5 TFLOPS；FP32 494.5 TFLOPS。<br>CPU：Intel Xeon 6767P；BF16 / FP16 255.29 TFLOPS；INT8 511.18 TOPS。<br>每个 HBM：144 GB；每个 DDR5：1536 GB；每个 NVMe SSD：256000 GB。<br>NIC：Mellanox ConnectX-7，400 Gb/s InfiniBand。<br>线色：蓝色 NVLink；绿色 HBM；灰色 PCIe / SSD；青色 NIC PCIe / InfiniBand；紫色 CPU PCIe / DDR5 / QPI（底部虚线）。',40,1590,2300,230)
path=Path(__file__).resolve().parent / 'H200Cluster-full-topology.drawio'
ET.indent(mx); ET.ElementTree(mx).write(path,encoding='utf-8',xml_declaration=True)
check=ET.parse(path); cells=check.findall('.//mxCell'); ids={c.get('id') for c in cells}
for c in cells:
    if c.get('edge')=='1': assert c.get('source') in ids and c.get('target') in ids
edges=[c for c in cells if c.get('edge')=='1']
assert len(edges)==59, len(edges)
print(str(path)); print('Validated: 46 hardware nodes, 59 links; UTF-8 draw.io XML')
