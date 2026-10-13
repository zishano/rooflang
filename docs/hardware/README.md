# H200Cluster topology

Complete topology based on `programs/presets/h200.py`: 46 hardware nodes and 59 links for one node, including the shared cluster IB switch. The topology shows device names and connections only. Device specifications, link bandwidth, latency, duplex settings, and multi-node replication rules are listed once in the footer; edge details are also available as draw.io tooltips.

- `H200Cluster-full-topology.drawio`: editable diagrams.net XML.
- `H200Cluster-full-topology.png`: 4800 × 3700 preview.

Regenerate from any working directory:

```bash
python docs/hardware/generate_h200_topology.py
python docs/hardware/render_h200_topology.py
```

The generator uses the Python standard library. The PNG renderer needs Pillow (included through the project matplotlib dependency) and `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`. Adjust the renderer font path for other platforms. Both scripts write beside themselves. Hardware parameters are explicitly encoded from the preset; changes to the preset require updating the generator.
