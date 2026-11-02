"""Generate an editable, fixed-coordinate draw.io version of the GB300 graph."""

from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree

OUT = Path(__file__).with_name("gb300_cluster_8gpu_fixed.drawio")

mxfile = Element("mxfile", host="app.diagrams.net", modified="2026-11-02T00:00:00.000Z")
diagram = SubElement(mxfile, "diagram", id="gb300-fixed", name="GB300Cluster-8GPU")
model = SubElement(diagram, "mxGraphModel", dx="1900", dy="980", grid="1", gridSize="10",
                   guides="1", tooltips="1", connect="1", arrows="1", fold="1",
                   page="1", pageScale="1", pageWidth="1900", pageHeight="980",
                   math="0", shadow="0")
root = SubElement(model, "root")
SubElement(root, "mxCell", id="0")
SubElement(root, "mxCell", id="1", parent="0")


def vertex(cid, value, x, y, w, h, style):
    c = SubElement(root, "mxCell", id=cid, value=value, style=style,
                   vertex="1", parent="1")
    SubElement(c, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), as_="geometry")
    return c


def edge(cid, points, color):
    # Absolute endpoints and explicit waypoints: no attached terminals and no
    # automatic router. Moving a box requires moving its line endpoints too.
    c = SubElement(root, "mxCell", id=cid, value="",
                   style=("edgeStyle=none;noEdgeStyle=1;rounded=0;curved=0;"
                          "html=1;endArrow=none;startArrow=none;"
                          f"strokeColor={color};strokeWidth=2;"),
                   edge="1", parent="1")
    geom = SubElement(c, "mxGeometry", relative="1", as_="geometry")
    for key, (x, y) in (("sourcePoint", points[0]), ("targetPoint", points[-1])):
        SubElement(geom, "mxPoint", x=str(x), y=str(y), as_=key)
    if len(points) > 2:
        arr = SubElement(geom, "Array", as_="points")
        for x, y in points[1:-1]:
            SubElement(arr, "mxPoint", x=str(x), y=str(y))
    return c


base = "rounded=1;whiteSpace=wrap;html=1;align=center;verticalAlign=middle;fontFamily=Microsoft YaHei;"
styles = {
    "gpu": base + "fillColor=#dbeafe;strokeColor=#2563eb;strokeWidth=1.5;fontStyle=1;",
    "memory": base + "fillColor=#dcfce7;strokeColor=#16a34a;strokeWidth=1.5;fontStyle=1;",
    "cpu": base + "fillColor=#f3e8ff;strokeColor=#9333ea;strokeWidth=1.5;fontStyle=1;",
    "nic": base + "fillColor=#cffafe;strokeColor=#0891b2;strokeWidth=1.5;fontStyle=1;",
    "ssd": base + "fillColor=#f1f5f9;strokeColor=#64748b;strokeWidth=1.5;fontStyle=1;",
    "fabric": base + "fillColor=#fef3c7;strokeColor=#d97706;strokeWidth=1.5;fontStyle=1;",
}

vertex("outer", "", 20, 90, 1860, 660,
       "rounded=1;whiteSpace=wrap;html=1;fillColor=#fbfdff;strokeColor=#94a3b8;dashed=1;dashPattern=6 5;")
vertex("tray0", "<b>Compute Tray 0：GPU 0–3</b>", 40, 175, 900, 555,
       "swimlane;rounded=1;html=1;horizontal=1;startSize=30;fillColor=#f8fbff;strokeColor=#93c5fd;strokeWidth=2;fontFamily=Microsoft YaHei;")
vertex("tray1", "<b>Compute Tray 1：GPU 4–7</b>", 960, 175, 900, 555,
       "swimlane;rounded=1;html=1;horizontal=1;startSize=30;fillColor=#f8fbff;strokeColor=#93c5fd;strokeWidth=2;fontFamily=Microsoft YaHei;")
vertex("title", "<b>GB300Cluster（nvl_scope=8）：单节点完整拓扑与跨节点互联</b>", 500, 20, 900, 35,
       "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=24;fontFamily=Microsoft YaHei;")
vertex("node_label", "<b>节点 n0：8-GPU NVLink 域</b>", 750, 95, 400, 25,
       "text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=16;fontFamily=Microsoft YaHei;")
vertex("nvs", "<b>逻辑第 5 代 NVSwitch Fabric</b><br>节点内全部GPU共享", 650, 125, 600, 40, styles["fabric"])
vertex("ibs", "<b>全集群共享 InfiniBand Switch</b><br>多节点时，各节点的所有NIC均连接到这里",
       665, 790, 570, 50, styles["fabric"])

xs = [75, 295, 515, 735, 995, 1215, 1435, 1655]
for i, x in enumerate(xs):
    vertex(f"g{i}", f"<b>GB300 GPU {i}</b>", x, 235, 150, 50, styles["gpu"])
    vertex(f"h{i}", f"<b>HBM3e {i}</b><br>288 GB", x, 320, 150, 50, styles["memory"])
    vertex(f"s{i}", f"<b>NVMe SSD {i}</b><br>256 TB", x, 560, 150, 50, styles["ssd"])
    vertex(f"i{i}", f"<b>ConnectX-8 NIC {i}</b>", x, 650, 150, 48, styles["nic"])

cpu_centers = [260, 700, 1180, 1620]
for i, cx in enumerate(cpu_centers):
    vertex(f"c{i}", f"<b>Grace CPU {i}</b>", cx - 75, 410, 150, 48, styles["cpu"])
    vertex(f"d{i}", f"<b>DRAM {i}</b><br>480 GB", cx - 75, 485, 150, 48, styles["memory"])

eid = 0

def e(points, color):
    global eid
    eid += 1
    edge(f"e{eid}", points, color)

# Each horizontal bus is a fixed line, with separate vertical taps.
e([(150, 215), (1730, 215)], "#2563eb")
e([(950, 165), (950, 215)], "#2563eb")
e([(150, 725), (1730, 725)], "#0e7490")
e([(950, 725), (950, 790)], "#0e7490")
for x in xs:
    e([(x+75, 215), (x+75, 235)], "#2563eb")
    e([(x+75, 285), (x+75, 320)], "#16a34a")
    e([(x+150, 260), (x+165, 260), (x+165, 674), (x+150, 674)], "#0891b2")
    e([(x+75, 698), (x+75, 725)], "#0e7490")

for i, cx in enumerate(cpu_centers):
    left, right = xs[2*i], xs[2*i+1]
    e([(left+150, 260), (left+150, 390), (cx, 390), (cx, 410)], "#9333ea")
    e([(right, 260), (right, 390), (cx, 390)], "#9333ea")
    e([(cx, 458), (cx, 485)], "#16a34a")
    e([(cx-75, 434), (left+75, 434), (left+75, 560)], "#64748b")
    e([(cx+75, 434), (right+75, 434), (right+75, 560)], "#64748b")
for left, right in ((260, 700), (1180, 1620)):
    e([(left+75, 422), (right-75, 422)], "#9333ea")

vertex("legend", ("<b>链路参数（GB/s）</b><br>GPU–HBM 8000｜GPU–NVSwitch 900｜"
                  "Grace–GPU / Grace–Grace 450｜GPU–NIC 128｜NIC–IB 100<br>"
                  "SSD→Grace 14｜Grace→SSD 7｜Grace–DRAM 384"),
       250, 875, 1400, 55,
       "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff7ed;strokeColor=#ea580c;align=center;fontFamily=Microsoft YaHei;")

# ElementTree names the XML attribute "as_" literally; draw.io expects "as".
for elem in root.iter():
    if "as_" in elem.attrib:
        elem.attrib["as"] = elem.attrib.pop("as_")

ElementTree(mxfile).write(OUT, encoding="utf-8", xml_declaration=True)
print(OUT)
