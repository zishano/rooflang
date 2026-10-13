# DSV4 Flash compute graph

`DSV4-Flash-compute-graph.drawio` is a single editable diagrams.net page with four sections: the full single-step decode model, a transformer layer, attention, and MoE. `DSV4-Flash-compute-graph.png` is a matching 5600 × 3480 preview.

The graph follows `programs/models/dsv4_flash/model.py` and `config.py`. Dashed arrows denote expansion, not computation dependencies. Structural fan-out, weights, and KV persistence barrier edges are summarized in the footer. This is a teaching diagram of the declaration phase, before device placement and parallel optimization.

Regenerate from the repository root:

```bash
python docs/models/generate_dsv4_flash_graph.py
```

Requires Pillow and `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`. All output files are written beside the script.
