from pathlib import Path

OUT = Path(__file__).with_name("gb300_cluster_8gpu_clean.svg")
PNG_OUT = Path(__file__).with_name("gb300_cluster_8gpu_clean.png")
W, H = 1900, 980


def esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
  <marker id="arrow" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="context-stroke"/></marker>
  <style>
    text {{ font-family: "Noto Sans CJK SC", "Microsoft YaHei", Arial, sans-serif; fill:#172033; }}
    .title {{ font-size:25px; font-weight:700; }} .subtitle {{ font-size:15px; fill:#475569; }}
    .group {{ font-size:17px; font-weight:700; }} .node {{ font-size:13px; font-weight:600; }}
    .small {{ font-size:11px; font-weight:400; }} .legend {{ font-size:12px; }}
    .bi {{ fill:none; stroke-width:2; }}
  </style>
</defs>
<rect width="100%" height="100%" fill="white"/>
<text x="950" y="40" text-anchor="middle" class="title">GB300Cluster（nvl_scope=8）：单节点完整拓扑与跨节点互联</text>
<text x="950" y="66" text-anchor="middle" class="subtitle">严格按 programs/presets/gb300.py 建模；同类设备逐一展开</text>
<rect x="20" y="90" width="1860" height="660" rx="16" fill="#fbfdff" stroke="#94a3b8" stroke-width="2" stroke-dasharray="7 6"/>
<text x="950" y="116" text-anchor="middle" class="group">节点 n0：8-GPU NVLink 域</text>
<rect x="40" y="175" width="900" height="555" rx="13" fill="#f8fbff" stroke="#93c5fd" stroke-width="2"/>
<rect x="960" y="175" width="900" height="555" rx="13" fill="#f8fbff" stroke="#93c5fd" stroke-width="2"/>
<text x="490" y="202" text-anchor="middle" class="group">Compute Tray 0：GPU 0–3</text>
<text x="1410" y="202" text-anchor="middle" class="group">Compute Tray 1：GPU 4–7</text>
''']


def line(x1, y1, x2, y2, color, dashed=False, width=2):
    dash = ' stroke-dasharray="6 5"' if dashed else ''
    parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" class="bi" stroke="{color}" stroke-width="{width}"{dash}/>')


def box(x, y, w, h, label, fill, stroke, sub=""):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    cy = y + h / 2 - (7 if sub else 0)
    parts.append(f'<text x="{x+w/2}" y="{cy}" text-anchor="middle" dominant-baseline="middle" class="node">{esc(label)}</text>')
    if sub:
        parts.append(f'<text x="{x+w/2}" y="{cy+18}" text-anchor="middle" dominant-baseline="middle" class="small">{esc(sub)}</text>')


xs = [75, 295, 515, 735, 995, 1215, 1435, 1655]
gpu_y, hbm_y, cpu_y, dram_y, ssd_y, nic_y = 235, 320, 410, 485, 560, 650

# Links are emitted before boxes so that nodes remain visually dominant.
# NVSwitch fan-out uses one horizontal bus, matching the H200 reference.
parts.append(f'<path d="M950,165 V205 H{xs[0]+75} H{xs[-1]+75}" fill="none" stroke="#2563eb" stroke-width="2"/>')
for x in xs:
    parts.append(f'<path d="M{x+75},205 V{gpu_y}" fill="none" stroke="#2563eb" stroke-width="2"/>')
    line(x + 75, gpu_y + 50, x + 75, hbm_y, "#16a34a")    # GPU–HBM
    # Route GPU–NIC around the memory/storage boxes so it cannot be mistaken
    # for a serial GPU→HBM→SSD→NIC path.
    parts.append(
        f'<path d="M{x+150},{gpu_y+25} H{x+165} V{nic_y+24} H{x+150}" '
        'class="bi" stroke="#0891b2"/>')

cpu_centers = [260, 700, 1180, 1620]
for ci, cx in enumerate(cpu_centers):
    left_x = xs[2 * ci]
    right_x = xs[2 * ci + 1]
    # One C2C bus joins the inner side of both GPUs to their Grace CPU.
    parts.append(
        f'<path d="M{left_x+150},{gpu_y+25} V390 H{cx} V{cpu_y}" '
        'class="bi" stroke="#9333ea"/>')
    parts.append(
        f'<path d="M{right_x},{gpu_y+25} V390 H{cx}" '
        'class="bi" stroke="#9333ea"/>')
    line(cx, cpu_y + 48, cx, dram_y, "#16a34a")
    parts.append(
        f'<path d="M{cx-75},{cpu_y+24} H{left_x+75} V{ssd_y}" '
        'class="bi" stroke="#64748b"/>')
    parts.append(
        f'<path d="M{cx+75},{cpu_y+24} H{right_x+75} V{ssd_y}" '
        'class="bi" stroke="#64748b"/>')

# Only CPUs inside the same 4-GPU compute tray are paired in code.
line(cpu_centers[0] + 75, cpu_y + 24, cpu_centers[1] - 75, cpu_y + 24, "#a855f7")
line(cpu_centers[2] + 75, cpu_y + 24, cpu_centers[3] - 75, cpu_y + 24, "#a855f7")

# Shared IB uses one horizontal collection bus.
parts.append(f'<path d="M{xs[0]+75},725 H{xs[-1]+75}" fill="none" stroke="#0e7490" stroke-width="2"/>')
for x in xs:
    parts.append(f'<path d="M{x+75},{nic_y+48} V725" fill="none" stroke="#0e7490" stroke-width="2"/>')
parts.append('<path d="M950,725 V790" fill="none" stroke="#0e7490" stroke-width="2"/>')

# Logical fabrics and components.
box(650, 125, 600, 40, "逻辑第 5 代 NVSwitch Fabric", "#fef3c7", "#d97706", "节点内全部 GPU 共享")
for i, x in enumerate(xs):
    box(x, gpu_y, 150, 50, f"GB300 GPU {i}", "#dbeafe", "#2563eb")
    box(x, hbm_y, 150, 50, f"HBM3e {i}", "#dcfce7", "#16a34a", "288 GB")
    box(x, ssd_y, 150, 50, f"NVMe SSD {i}", "#f1f5f9", "#64748b", "256 TB")
    box(x, nic_y, 150, 48, f"ConnectX-8 NIC {i}", "#cffafe", "#0891b2")

for i, cx in enumerate(cpu_centers):
    box(cx - 75, cpu_y, 150, 48, f"Grace CPU {i}", "#f3e8ff", "#9333ea")
    box(cx - 75, dram_y, 150, 48, f"DRAM {i}", "#dcfce7", "#16a34a", "480 GB")

box(665, 790, 570, 50, "全集群共享 InfiniBand Switch", "#fef3c7", "#d97706", "多节点时，各节点的所有 NIC 均连接到这里")

# Legend, with one representative specification per link class.
legend = [
    ("#16a34a", "GPU↔HBM：8000 GB/s，α=0.5 μs，非全双工"),
    ("#2563eb", "GPU↔NVSwitch：900 GB/s，α=0.5 μs，全双工"),
    ("#9333ea", "Grace↔GPU / Grace↔Grace：NVLink-C2C 450 GB/s，α=0.5 μs，全双工"),
    ("#0891b2", "GPU↔NIC：PCIe 128 GB/s，α=0.5 μs，全双工"),
    ("#64748b", "SSD→Grace 14、Grace→SSD 7 GB/s，α=50 μs，全双工"),
    ("#0e7490", "NIC↔IB：100 GB/s（800 Gb/s），α=1.0 μs，全双工"),
]
for idx, (color, label) in enumerate(legend):
    col, row = idx % 3, idx // 3
    x, y = 70 + col * 610, 885 + row * 42
    parts.append(f'<line x1="{x}" y1="{y}" x2="{x+45}" y2="{y}" stroke="{color}" stroke-width="4"/>')
    parts.append(f'<text x="{x+58}" y="{y+4}" class="legend">{esc(label)}</text>')

parts.append('<text x="950" y="970" text-anchor="middle" class="small">说明：图中没有 PCIe Switch 节点，因为当前 RoofLang GB300 代码直接建立 GPU–NIC 与 SSD–Grace 链路。</text>')
parts.append('</svg>')
OUT.write_text("\n".join(parts), encoding="utf-8")

# Produce a directly viewable PNG with the same fixed layout. Keeping the SVG
# as the canonical vector artifact avoids depending on a browser renderer.
from PIL import Image, ImageDraw, ImageFont
from math import atan2, cos, sin, pi

img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)
regular = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
bold = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
fonts = {
    "title": ImageFont.truetype(bold, 25), "subtitle": ImageFont.truetype(regular, 15),
    "group": ImageFont.truetype(bold, 17), "node": ImageFont.truetype(bold, 13),
    "small": ImageFont.truetype(regular, 11), "legend": ImageFont.truetype(regular, 12),
}


def txt(x, y, value, font, anchor="mm", fill="#172033"):
    d.text((x, y), value, font=fonts[font], fill=fill, anchor=anchor, align="center")


def arrow(x1, y1, x2, y2, color, width=2):
    d.line((x1, y1, x2, y2), fill=color, width=width)


def poly_arrow(points, color, width=2):
    d.line(points, fill=color, width=width, joint="curve")


def pbox(x, y, w, h, label, fill, stroke, sub=""):
    d.rounded_rectangle((x, y, x + w, y + h), radius=7, fill=fill, outline=stroke, width=2)
    txt(x + w / 2, y + h / 2 - (7 if sub else 0), label, "node")
    if sub:
        txt(x + w / 2, y + h / 2 + 12, sub, "small")


txt(950, 35, "GB300Cluster（nvl_scope=8）：单节点完整拓扑与跨节点互联", "title")
txt(950, 64, "严格按 programs/presets/gb300.py 建模；同类设备逐一展开", "subtitle", fill="#475569")
d.rounded_rectangle((20, 90, 1880, 750), radius=16, fill="#fbfdff", outline="#94a3b8", width=2)
txt(950, 112, "节点 n0：8-GPU NVLink 域", "group")
d.rounded_rectangle((40, 175, 940, 730), radius=13, fill="#f8fbff", outline="#93c5fd", width=2)
d.rounded_rectangle((960, 175, 1860, 730), radius=13, fill="#f8fbff", outline="#93c5fd", width=2)
txt(490, 198, "Compute Tray 0：GPU 0–3", "group")
txt(1410, 198, "Compute Tray 1：GPU 4–7", "group")

arrow(950, 165, 950, 205, "#2563eb")
arrow(xs[0] + 75, 205, xs[-1] + 75, 205, "#2563eb")
for x in xs:
    arrow(x + 75, 205, x + 75, gpu_y, "#2563eb")
    arrow(x + 75, gpu_y + 50, x + 75, hbm_y, "#16a34a")
    poly_arrow([(x + 150, gpu_y + 25), (x + 165, gpu_y + 25),
                (x + 165, nic_y + 24), (x + 150, nic_y + 24)], "#0891b2")
for ci, cx in enumerate(cpu_centers):
    left_x, right_x = xs[2 * ci], xs[2 * ci + 1]
    poly_arrow([(left_x + 150, gpu_y + 25), (left_x + 150, 390),
                (cx, 390), (cx, cpu_y)], "#9333ea")
    poly_arrow([(right_x, gpu_y + 25), (right_x, 390),
                (cx, 390)], "#9333ea")
    arrow(cx, cpu_y + 48, cx, dram_y, "#16a34a")
    poly_arrow([(cx - 75, cpu_y + 24), (left_x + 75, cpu_y + 24),
                (left_x + 75, ssd_y)], "#64748b")
    poly_arrow([(cx + 75, cpu_y + 24), (right_x + 75, cpu_y + 24),
                (right_x + 75, ssd_y)], "#64748b")
arrow(cpu_centers[0] + 75, cpu_y + 24, cpu_centers[1] - 75, cpu_y + 24, "#a855f7")
arrow(cpu_centers[2] + 75, cpu_y + 24, cpu_centers[3] - 75, cpu_y + 24, "#a855f7")
arrow(xs[0] + 75, 725, xs[-1] + 75, 725, "#0e7490")
for x in xs:
    arrow(x + 75, nic_y + 48, x + 75, 725, "#0e7490")
arrow(950, 725, 950, 790, "#0e7490")

pbox(650, 125, 600, 40, "逻辑第 5 代 NVSwitch Fabric", "#fef3c7", "#d97706", "节点内全部 GPU 共享")
for i, x in enumerate(xs):
    pbox(x, gpu_y, 150, 50, f"GB300 GPU {i}", "#dbeafe", "#2563eb")
    pbox(x, hbm_y, 150, 50, f"HBM3e {i}", "#dcfce7", "#16a34a", "288 GB")
    pbox(x, ssd_y, 150, 50, f"NVMe SSD {i}", "#f1f5f9", "#64748b", "256 TB")
    pbox(x, nic_y, 150, 48, f"ConnectX-8 NIC {i}", "#cffafe", "#0891b2")
for i, cx in enumerate(cpu_centers):
    pbox(cx - 75, cpu_y, 150, 48, f"Grace CPU {i}", "#f3e8ff", "#9333ea")
    pbox(cx - 75, dram_y, 150, 48, f"DRAM {i}", "#dcfce7", "#16a34a", "480 GB")
pbox(665, 790, 570, 50, "全集群共享 InfiniBand Switch", "#fef3c7", "#d97706", "多节点时，各节点的所有 NIC 均连接到这里")

for idx, (color, label) in enumerate(legend):
    col, row = idx % 3, idx // 3
    x, y = 70 + col * 610, 885 + row * 42
    d.line((x, y, x + 45, y), fill=color, width=4)
    txt(x + 58, y, label, "legend", anchor="lm")
txt(950, 968, "说明：图中没有 PCIe Switch 节点，因为当前 RoofLang GB300 代码直接建立 GPU–NIC 与 SSD–Grace 链路。", "small")
img.save(PNG_OUT)
print(OUT)
print(PNG_OUT)
