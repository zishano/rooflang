# DSV4 Flash compute graph

`DSV4-Flash-compute-graph.drawio` is a single editable diagrams.net page with four sections: the full single-step decode model, a transformer layer, attention, and MoE. `DSV4-Flash-compute-graph.png` is a matching 5600 × 3480 preview.

The graph follows `programs/models/dsv4_flash/model.py` and `config.py`. Dashed arrows denote expansion, not computation dependencies. Structural fan-out, weights, and KV persistence barrier edges are summarized in the footer. This is a teaching diagram of the declaration phase, before device placement and parallel optimization.

Regenerate from the repository root:

```bash
python docs/models/generate_dsv4_flash_graph.py
```

Requires Pillow and `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`. All output files are written beside the script.

## Prefill comparison

`DSV4-Flash-prefill-compute-graph.drawio` and its PNG show the prefill variant in the same four-section format. The expanded attention panel represents a ratio=4 layer, with notes for ratio=0 and ratio=128. It includes the three-way input bridge, window/compressed KV construction, index cache construction, a shared cache-lifetime barrier, and the final-token slice. The decode files remain available for comparison.

```bash
python docs/models/generate_dsv4_flash_prefill_graph.py
```

Validation: built the repository prefill graph with batch size 1 and sequence length 128; checked layers with ratios 0, 4, and 128, the last-token slice, and the shared persistence barrier. This validates the illustrated structure, not numerical inference or performance.

## Optimized Decode and Placement

`DSV4-Flash-optimized-decode.drawio` contains three editable pages, with PNG previews:

1. `DSV4-Flash-optimized-operators.png`: a DP group in layer 2 and the shared EP MoE flow.
2. `DSV4-Flash-optimized-placement.png`: GPU and memory mapping.
3. `DSV4-Flash-optimized-process.png`: graph transformation/placement versus simulator execution order.

Example: batch 8, prefix 8192, DP 4, CP 2, EP 8, PP [43], one 8-GPU H200 node. The actual graph was constructed and optimized; key shapes, assignments, communication neighbors and expert ownership were checked. See `DSV4-Flash-optimized-validation.json`. Timing simulation was not run. The drawings summarize dependencies and repeated nodes, not a measured timeline.

Regenerate with `python docs/models/generate_dsv4_flash_optimized.py` (Pillow and Noto CJK font required).

Each optimized diagram is also available as an independent single-page draw.io file, with the same editable contents and geometry as the corresponding page in the combined file:

- `DSV4-Flash-optimized-operators.drawio`
- `DSV4-Flash-optimized-placement.drawio`
- `DSV4-Flash-optimized-process.drawio`

The generator exports and validates all three files alongside the combined document.
