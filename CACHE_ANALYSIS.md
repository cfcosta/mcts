# Cache behavior of the search (baseline analysis)

Measured 2026-07-27 against the `initial` benchmark baseline, before any
optimization work. Everything here describes the *current* implementation;
treat it as the "why" behind the optimization roadmap, and re-measure after
structural changes.

Machine: AMD Ryzen 9 7950X3D (Zen 4) — 32 KiB L1d + 1 MiB L2 per core,
128 MiB L3 (3D V-cache). The unusually large L3 matters: it hides capacity
misses that would dominate on a typical 8–32 MiB CPU, so the numbers below
are a *best case* for the current layout.

Method: a scratch binary depending on this crate ran fixed workloads under
`perf stat` / `perf record` (hardware counters) and Valgrind's cachegrind
(simulated 32 KiB L1 + per-function attribution), plus a counting global
allocator and `std::mem::offset_of` for layout. The probe adds ~1.4%
overhead (its per-search times match the Criterion baseline within that).

## Verdict

The data layout is cache-hostile by construction — fat array-of-structs
nodes, three heap allocations per node, and a `Vec` clone per
`get_legal_actions` call — but at bench-sized trees most of the damage is
absorbed by this CPU's L2/L3 and by hardware prefetch. The measurable
symptoms today are a low IPC (~2 on a core capable of ~5), an L1d that is
far too small for the per-node footprint (8.3% L1d miss rate on the deep
chain despite a 41 KiB node array), the UCB scan producing ~18% of all L1d
read misses from ~6.5% of instructions, and the allocator consuming ~20% of
cycles. Scaling the tree to RAM size (350 MiB at 100k iterations) costs
only ~15% more cycles per iteration *on this machine* thanks to the V-cache
— expect much worse on ordinary hardware.

## Data layout facts

```
Node<UltimateTicTacToe>: 144 B (2.25 cache lines)   Node<TicTacToe>: 112 B
  parent      offset   0..16                          state    16..56
  state       offset  16..88   (includes a Vec)       children 56..80
  children    offset  88..112  (separate heap alloc)  n, q     88..96, 96..104
  reward_sum  offset 112..120
  n           offset 120..128  <- UCB reads these two:
  q           offset 128..136  <- they sit on DIFFERENT cache lines
```

Per UTTT node the tree stores **three separate allocations**:

| allocation | bytes | note |
|---|---|---|
| the node in the arena | 144 | contiguous `Vec<Node>` |
| `state.legal_actions` | 243 | `Vec<(u8,u8,u8)>` with capacity 81, kept forever |
| `node.children` | ~72 | `Vec<usize>`, one per *expanded* node |

Measured tree growth (UTTT, c = 1.41): **8.6–8.8 nodes created per
iteration** (expansion creates every child at once), ~4 KiB of new heap per
iteration. Tree sizes: 1k iters → 8.8k nodes / 6.6 MiB RSS; 10k → 87k
nodes / 38 MiB; 100k → 855k nodes / **350 MiB** (117 MiB nodes + 198 MiB
of capacity-81 `legal_actions` buffers + 10 MiB children vecs). Tic-Tac-Toe
is far smaller (1.3 nodes/iteration; 10k iters → 13k nodes, 1.4 MiB).

Two useful invariants were verified on every tree: children ids are always
a contiguous range (they are pushed consecutively during expansion), and
`n`/`q`/`reward_sum` are the only fields backprop touches.

## Measured behavior

`perf stat` (user-space, whole process):

| workload | working set | IPC | L1d miss | LLC miss | note |
|---|---|---|---|---|---|
| UTTT search, 1k iters ×20 | ~6 MiB/search | 2.11 | 0.89% | ~0 | fits L2+L3 |
| UTTT search, 100k iters | ~350 MiB | 1.93 | 1.71% | 3.3M (~33/iter) | V-cache absorbs most |
| TTT search, 10k iters ×20 | ~1.6 MiB | 2.79 | 1.88% | ~0 | |
| chain 512 deep, 2k iters | **41 KiB** | 2.31 | **8.31%** | ~0 | select+backprop walk |
| wide 512, 1k iters | ~49 KiB | 3.43 | 4.53% | ~0 | UCB scan stride |

The deep chain is the smoking gun: 513 nodes × 80 B ≈ 41 KiB plus 513
scattered children-`Vec` allocations exceed the 32 KiB L1d, so every
select-down/backprop-up pass re-streams the whole path and misses L1 on
most levels — with the entire tree just 41 KiB. Per-node line footprint
during selection is ~3–4 lines (node + its children vec + each child's
`n`/`q` on 1–2 lines + a re-read of the parent's `n` per UCB call).

Attribution (cachegrind D1, UTTT 1k×10; percentages of all D1 read misses):
`Node::get_best_child` **17.9%** (from 6.5% of instructions), glibc
`memcpy` 20.8% (cloning states into the arena; also 66% of D1 *write*
misses — expansion writes 144 B × ~8.7 nodes per iteration plus arena
regrowth), `malloc`/`free`/`_int_malloc` ~8%, the rest spread across
rollout state stepping.

Cycle profile (`perf record`, UTTT 100k iters): `step` 26%,
`get_best_child` 19%, allocator ~23% (alloc 19% + free 4.4%), `search`
(select/expand/backprop) 16%, `ln` 3.4%. Allocator calls run at **~135 per
iteration** (measured; ~120 of them are the rollout's per-ply
`get_legal_actions` clone + the fresh capacity-81 vec built by every
`step`). Branch misses cost roughly another fifth of cycles (55M misses
over 3.6G cycles at 100k iters) — inherent to random rollouts, but
magnified by the branchy scans. dTLB misses are negligible (hugepages).

## Ranked opportunities (cache-focused)

1. **Stop storing `legal_actions` inside the state / stop cloning it.**
   Kills ~120 of 135 allocs/iteration (rollout) and 243 B/node of dead
   capacity (57% of tree heap). Biggest single lever; also shrinks
   `Node<UTTT>` toward ~1 line.
2. **Split hot from cold node data (SoA or a hot struct).** Selection needs
   only `{n, q, first_child, len}`; with `u32`/`f32` that is 16 B — four
   children per cache line instead of one child per ~1.5 lines, and `n`/`q`
   never straddle lines again. Backprop needs only `{parent, n, q,
   reward_sum}`.
3. **Replace `children: Vec<usize>` with `first_child: u32 + len: u8`.**
   Valid today (contiguity verified); removes one heap allocation and one
   pointer chase per node, ~72 B/node, and the per-level indirection in
   select.
4. **Hoist `parent.n` out of the per-child UCB call** (pass it down; also
   precompute `c * ln(parent_n).sqrt()… once per scan`) and avoid
   `max_by`'s re-evaluation of the running max — halves the loads and the
   `ln` count in the hottest loop.
5. **`Vec::with_capacity` / `reserve` for the arena** (growth doublings
   memmove the entire arena — ~2× the final 117 MiB copied over a 100k
   search — and showed up as 66% of D1 write misses together with node
   cloning).
6. **Reuse a scratch state + action buffer in rollouts** instead of
   `clone`-per-ply, eliminating the remaining allocator traffic.
7. **Tree reuse across moves** (advance the root) — turns the per-move
   rebuild in self-play into an incremental update; changes observable
   tree shape, so it must be validated against the behavioral tests, not
   the tree-invariant internals.

Re-measure after each change: `cargo bench --bench mcts -- --baseline
initial`, and re-run the perf/cachegrind probes for anything touching node
layout. On machines with normal L3 sizes, expect items 1–3 to matter even
more than they do here.
