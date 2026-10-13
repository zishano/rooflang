# H200Cluster topology

Hierarchical cluster topology based on `programs/presets/h200.py`: two example nodes share one global IB switch. Each node shows device lanes 0 and 7; lanes 1–6 and additional nodes are represented by ellipses. CPU 0 connects lanes 0–3 and CPU 1 connects lanes 4–7; omitted lane links are explained in the diagram. The topology shows device names and connections only. Device specifications, link bandwidth, latency, duplex settings, and multi-node replication rules are listed once in the footer; edge details are also available as draw.io tooltips.

- `H200Cluster-full-topology.drawio`: editable diagrams.net XML.
- `H200Cluster-full-topology.png`: 4800 × 3300 preview.

Regenerate from any working directory:

```bash
python docs/hardware/generate_h200_topology.py
python docs/hardware/render_h200_topology.py
```

The generator uses the Python standard library. The PNG renderer needs Pillow (included through the project matplotlib dependency) and `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`. Adjust the renderer font path for other platforms. Both scripts write beside themselves. Hardware parameters are explicitly encoded from the preset; changes to the preset require updating the generator.

## H200SuperChip

`H200SuperChip-topology.drawio` and `H200SuperChip-topology.png` show the aggregated eight-GPU node. All 9 hardware components and 8 connections were checked against an instantiated `H200SuperChip` from `programs/presets/h200.py`. The diagram has no NVSwitch or QPI, matching the aggregation in code. Device and link parameters are listed once below the topology.

Regenerate with Pillow and the same Noto CJK font used above:

```bash
python docs/hardware/generate_h200_superchip_topology.py
```
