# DSV4 Flash: Find a DCP Design Faster Than DCP=1

## Mission and fixed target

Autonomously implement, evaluate and refine coordinated parallel algorithms
until a correctness-passing DCP>1 design beats DCP=1 on complete decode
latency. Completing a topology sweep or reporting another unsuccessful
combination does not complete this task. Continue with new hypotheses based
on the measured bottleneck. Gains must be demonstrated, never assumed or
manufactured to satisfy this objective.

Fix the primary target: dsv4_flash, H200 x64 (eight GPUs per node), batch=64,
initial context=65536, EP=64, PP=1. Time four consecutive decode steps including
the ratio-4 compression boundary. Never simulate more than four steps per
evaluation; separate validation and paired replays may each use up to four.
Do not change batch, context, model, precision, expert routing, selected-token
count, GPU count or hardware costs to obtain a primary-target win. Other
workloads may be labeled diagnostics but cannot satisfy this objective.

Use explicit pipeline, request-DP, attention-DCP and expert-EP memberships
with stable GPU IDs. Do not assume DP * CP == EP as a universal constraint.
Record rank maps, request ownership, parent/overlap relations, replica scope,
collective participants and H200 links. Keep the EP group at the same 64 GPUs.
Start with DCP=2/4/8 node-local groups; explore other legal groups when a
concrete hypothesis justifies their cost. A winning DCP>1 design must actually
partition live attention KV work across ranks, not rename a DCP=1 path.

Preserve preexisting workspace changes and other agents. Do not implicitly
restore the deleted layer-selective DCP implementation as a search axis.

## Prior evidence

Previous runs:
- run-20261027T015718Z-fbpQvL: 12 topology-only candidates.
- run-20261027T033002Z-h0H7lR: 27 candidates including owner-sharded Index-K
  and hierarchical LSE merge; none beat DCP=1.

Historical four-step totals in microseconds:
- DCP=1 replicated/flat: 10971.903.
- DCP=2 node-local replicated/flat: 12165.588; combined: 12235.194.
- DCP=8 node-local: flat 12713.304; sharded-only 12739.693;
  hierarchical-only unchanged; combined 12739.693.
- DCP=16 contiguous: flat 16809.248; sharded-only 18205.960;
  hierarchical-only 20563.300; combined 21960.012.

These are historical observations, not current paired baseline measurements.
The previous sharded path gathers local candidate IDs/scores at one GPU,
selects global top-k there, then broadcasts IDs. The previous hierarchy adds
node-local merge before root merge. Merely changing their group sizes is not
a new algorithm. Index-K memory savings alone are not latency improvements.
Historical run folders may not be mounted: inspect available inputs, record
missing provenance, and reproduce essential controls rather than inventing
their contents. Do not assume prior experimental code is already merged.

## Fair controls and bottleneck diagnosis

1. Snapshot source hashes/configurations and audit the current DCP graph,
   state update, top-k, projection placement, communication, scheduling and
   terminal dependencies before editing.
2. Preserve a selectable original algorithm as B0. Run DCP=1 and a matched
   DCP=2 control with identical full-model measurement boundaries, simulator
   mode, inputs and hardware.
3. Profile representative layers and the full graph. Record FLOPs/dtype,
   input/weight/output bytes, placement, compute/memory/communication costs,
   dependency waits and critical-path contribution. Separate ordinary steps
   from compression-boundary steps.
4. Distinguish summed rank busy time from end-to-end latency. Overlapping
   kernel times cannot be added to explain wall latency. Use paired graph
   traces and single-change ablations to attribute changes.
5. Before implementation, estimate each hypothesis's break-even condition:
   useful work saved versus added communication, synchronization and
   materialization on the critical path. State assumptions and uncertainty.
   Failed candidates or summed traffic are not proof of global impossibility.

Maintain B0 (preserved algorithm) and B0+ (DCP=1 with every applicable general
optimization and comparable tuning). Apply shared modeling fixes to both
controls and candidates. Beating B0 but not B0+ is progress, not completion.
Explain why any proposed optimization genuinely cannot apply to DCP=1.

## New coordinated algorithm families

These are hypotheses, not promised improvements. Prioritize using measured
bottlenecks. Implement explicit graph semantics, placement, costs and an
independent reference. Propose additional families when profiling suggests
them; this list is not a closed search space.

### A. Distributed exact top-k and owner-directed execution

Replace centralized candidate gathering with node-local/tree selection,
request-distributed roots or exact pruning with a proven bound. Reduce
candidate sets before slow links. Couple selection to KV ownership so owners
receive only their required IDs when semantics permit, avoiding unnecessary
full-ID broadcasts. Charge queries, IDs, scores, intermediate selection,
routing, synchronization and updates. Prove global top-k equivalence including
deterministic ties; local top-k alone is not global top-k.

### B. Attention merge and downstream output placement

Compare request/head-distributed roots, balanced trees and streamed stable
softmax merge against the single-root gather. Coordinate final output
ownership with output projection, residual/norm and EP dispatch so a removed
gather is not followed by equivalent redistribution. Model all output/LSE
transfers and dependencies. Use reduce-scatter only when its algebra and
destination layout are correct; plain summation of locally normalized
attention outputs is invalid.

### C. Projection placement and attention parallelism

Audit duplicated query/index projection, normalization and activation movement
when DCP grows. Compare computing shared quantities once per request and
distributing them against redundant local work. Coordinate request/head
placement with KV lookup and output merge. Preserve tensor shapes, projection
work, residuals and expert routes; include layout conversions. Apply every
applicable generic projection optimization to B0+.

### D. Fine-grained execution and communication overlap

Chunk independent requests or attention work so query distribution, lookup,
local attention and merge can progress without unnecessary whole-group
barriers. Pipeline independent requests, not future decode tokens before
their dependencies are ready. Tune chunk size, placement and buffers jointly.
Demonstrate overlap in actual simulator event traces with resource contention
and true dependencies. Never subtract costs, drop live edges or relabel
sequential costs as overlap.

### E. Hybrid Index-K replication and exact selection

Compare per-node/selective-node replicas and owner sharding paired with
distributed selection and local execution. Jointly choose replica scope,
updates, selection roots and query/ID routing. Count all state memory and
append/update traffic. Invariant state can be reused across steps; scores,
top-k and attention depend on the new query and cannot be reused without
proof. Host graph-reuse speed is separate from simulated GPU latency.

### F. KV access and intermediate materialization

Compare moving selected KV to request owners against moving queries/partials
to KV owners. Consider fused lookup, local attention and online LSE only if
the allowed interfaces can faithfully model them. Count selected rows,
noncontiguous reads, duplicates, buffer lifetimes and outputs. Fusion may
remove genuine intermediate HBM traffic, not required FLOPs, source reads or
communication. Do not invent bandwidth, launch constants or faster kernel
costs. Coordinate EP placement/dispatch when useful while preserving expert
membership/routes/work; apply generic EP improvements to B0+ too.

## Continuous search and persistence

There is no task-imposed total candidate, iteration or wall-time limit.
The old 256-candidate overall cap is removed. The four-step evaluation cap
remains. External user stops, provider budgets, process limits and filesystem
permissions still apply; these instructions do not override them.

For every round:
1. Update bottleneck evidence and choose a new falsifiable hypothesis.
   Record expected mechanism, cost estimate and rejection criteria.
2. Implement the minimum complete variant, test semantics, then evaluate it
   at fixed topology before combining it with other changes.
3. Evaluate coordinated variants and B0/B0+ on identical inputs. Use small
   candidate batches to bound resource use, not as a termination limit.
   Separate results by source revision; never compare timings produced by
   different operator costs or accounting as if they were paired.
4. On failure, inspect the critical path and revise the algorithm or try
   another family. A completed grid, batch or family list is not a reason to
   finish when the objective remains unmet.
5. Persist progress after each candidate/round and continue until verified
   success or a documented external blocker.

Deduplicate by full configuration, source digest and seed. Rerun identical
candidates only for validation/replay. Layer/micrograph screening is allowed,
but only full-model four-step results can qualify. Keep incumbents, failures
and the pending queue so interruption does not require restarting the search.
A no-gain progress report is not a successful final result. If available
modeling capabilities or write permissions prevent meaningful further work,
document the exact missing capability and next required change. Never
fabricate a win or cycle through identical failures indefinitely.

## Correctness and modeling gates

- Maintain absolute window/compressed KV token-to-rank/local-row maps, ring
  overwrite, append, compression accumulation, Index-K ownership/replicas,
  global top-k localization and state dependencies across four steps.
- Execute production operators on small encoded tensors against an
  independent dense reference. Cover ties, empty/uneven shards, window wrap,
  compression boundaries and node-local/cross-node placement. Keep appropriate
  existing dtype tolerances fixed; do not loosen them to pass.
- Validate every changed projection/routing/fusion boundary. Do not inject
  reference outputs after the operation being optimized. Preserve stable
  per-head output/LSE merging and all required model outputs.
- Count weights, KV, Index-K, activations, selection/collective scratch and
  overlapping buffer lifetimes in per-GPU peak HBM. Charge communication to
  actual declared groups and physical paths.
- Preserve precision, model/config, useful work and selected-token counts.
  No free collectives, omitted updates, missing dependencies, fabricated byte
  savings or baseline-only penalties.
- State encoded-dataflow validation limits. Learned compression/quantization,
  pretrained logits and prefill-to-decode conversion remain unvalidated
  unless actually checked. Report preload/layout conversion separately and
  consistently across controls and candidates.

## Completion and artifacts

Success requires a genuine DCP>1 design faster than both B0 and B0+ at the
fixed target, passing numerical and per-GPU memory gates. Improvement must
exceed measured replay/precision uncertainty; report absolute and percentage
changes, ordinary-step and compression-step times, and any regressions.
Repeat the actual DCP>1 winner and both controls on the same final revision
and inputs. B0-versus-B0 replay cannot validate a DCP>1 winner. If execution
is nondeterministic, report repeated measurements and uncertainty.

Supply single-change and combined ablations explaining the improvement.
Memory-only savings or wins against a slower DCP>1 parent are useful evidence
but do not satisfy this objective.

Persist under /workspace/artifacts:
- checkpoint.json: status, round, code digest, controls/incumbent, completed
  IDs, pending hypotheses, issues and an implemented exact resume command.
- hypotheses.jsonl: mechanisms, predictions, results, rejection reasons and
  next experiments.
- candidates.jsonl/csv: exact topology/algorithm choices, hashes, seed,
  four step times, total, correctness, per-GPU memory, DCP/EP traffic,
  synchronization, scratch and failures.
- paired.jsonl, ablations.csv and representative traces: actual winner versus
  B0/B0+ and causal accounting.
- report.md and source patch/new files: reproducible commands, changes,
  verified gain or current gap, limitations, failures and outstanding work.

## Scope

Writable source: programs/models/dsv4_flash/optimization.py, dcp_topology.py,
dcp_graph.py, dcp_ops.py, dcp_validation.py, dcp_reuse.py;
programs/experiments/; tests/unit_tests/programs/. Other source is read-only.
Use these interfaces for custom operators and graphs. Do not bypass mounts
or modify shared simulator/hardware costs. If a hypothesis requires unsupported
resource/scheduling behavior, document the precise limitation and continue
supported hypotheses. Checkpoint a blocker only when meaningful remaining
work depends on the missing capability.

Use the existing Python environment and focused tests. Preserve unrelated
models, other agent folders and read-only results. Do not weaken tests.
Completion requires evidence; no instruction can guarantee that a faster
DCP design exists under this fixed workload and simulator.
