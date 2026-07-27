# Autoresearch: MCTS throughput and cache locality

## Objective
Improve the real performance of the generic MCTS implementation and bundled game states. Focus on vectorization opportunities, cache-local data layouts, fewer allocations and pointer chases, and less work in hot loops while preserving all observable behavior.

## Metrics
- **Primary**: `geomean_ns` (ns, lower is better) — geometric mean of six core workloads: empty/midgame end-to-end search for both games plus deep and wide tree operations. Each contributes by relative speedup rather than absolute duration.
- **Secondary**: `ttt_empty_ns`, `ttt_mid_ns`, `uttt_empty_ns`, `uttt_mid_ns`, `deep_chain_ns`, `wide_512_ns`, `uttt_rollout_ns`, `uttt_step_ns`.

Rollout and state-step microbenchmarks remain secondary diagnostics, but are excluded from the primary because unchanged reruns showed the nanosecond step metric swinging ~25% and overpowering otherwise consistent search results. Periodically run broader benchmark groups when a structural change may expose a tradeoff.

## How to Run
`./.auto/measure.sh` — enters a Nix shell providing `cargo` and `rustc`, runs the fixed-work harness pinned to CPU 6, and emits `METRIC name=value` lines. The harness mirrors the corresponding cases in `benches/mcts.rs` but uses nine batched medians because hard-coded Criterion measurement durations made an autonomous iteration take over 80 seconds. CPU 6 was selected after CPU 0's SMT sibling became busy and unchanged reruns drifted 6-13%.

`.auto/checks.sh` runs automatically after each successful measurement and executes the complete test suite with Cargo and Rust from Nix.

## Files in Scope
- `src/mcts/mod.rs` — search, selection, expansion, simulation, and backpropagation.
- `src/mcts/node.rs` — node representation and UCB child selection.
- `src/mcts/arena.rs` — arena representation and allocation strategy.
- `src/state.rs` — generic state interface, only when a broadly useful optimization requires it.
- `src/games/tic_tac_toe.rs` — bundled Tic-Tac-Toe state hot paths and layout.
- `src/games/ultimate_tic_tac_toe.rs` — bundled Ultimate Tic-Tac-Toe state hot paths and layout.
- `src/lib.rs` — exports, only if required by an internal redesign.
- `Cargo.toml` / `Cargo.lock` — only for justified production changes; avoid new dependencies.

## Off Limits
- Do not modify `benches/`, `tests/`, Criterion settings, or correctness thresholds to manufacture improvements.
- Do not special-case benchmark inputs or inspect whether code is running under a benchmark.
- Do not remove required work, reduce search iteration counts, weaken randomness, alter reward/search semantics, or rely on dead-code elimination.
- Do not optimize only the measured cases at the expense of realistic unmeasured inputs.
- Keep `.auto/measure.sh` and `.auto/bench/` workloads fixed unless correcting a measurement flaw; metric changes require a new experiment config.

## Constraints
- Every kept experiment must pass the full existing test suite.
- Preserve public behavior and the exact tree invariants pinned by tests, including one all-at-once expansion per selected node and current panic contracts.
- Preserve uniform random action/child choice where the implementation currently requires it.
- Use Cargo and Rust supplied by Nix (`nix-shell -p cargo rustc`).
- Prefer stable Rust and portable optimizations. No unsafe code without a compelling measured win and explicit invariant documentation.
- Treat sub-percent changes cautiously and remeasure noisy results.
- Do not overfit or cheat on benchmarks.

## What's Been Tried
- Established the original eight-workload Criterion baseline, then replaced only the timing harness because hard-coded group durations made each loop take ~82 seconds. The fixed-work harness retains the same positions, iteration counts, tree shapes, rollout, and step workloads.
- **Kept:** direct one-pass UCB scoring with a single-child fast path. This removed repeated `max_by` score evaluation and hoisted parent `ln`, improving the aggregate ~32% and wide-512 ~71%.
- **Kept:** removed cached legal-action `Vec`s from both bundled states. TTT is now a 6-byte `Copy` state and UTTT a 44-byte `Copy` state; UTTT stores only next-board/occupied metadata. This removed state clone/step allocation and retained capacity-81 buffers.
- **Kept:** optional allocation-free uniformly random action selection and in-place stepping for rollouts. Bundled states avoid per-ply vectors/state returns while external State implementations retain compatible defaults.
- **Kept:** a concrete rejection-corrected Lemire sampler replaced large generic `gen_range` monomorphizations. A shared 4.5 KiB compile-time table maps `(9-bit empty mask, uniform rank)` directly to packed row/column coordinates for both games; UTTT fallback bounds use its exact occupied-cell count rather than a redundant nine-board count pass.
- **Kept:** 512-byte compile-time win lookup tables for both games and empty-bit iteration for UTTT legal-action generation. Sharing the win table hurt code locality, so the duplicate per-module tables are intentional.
- **Kept:** per-search inverse-sqrt and selectively amortized sqrt-log lookup tables remove division/sqrt/log from UCB scans; tables are built only after a branching expansion. Combined with the one-pass scan, wide-512 is ~88% faster than baseline.
- **Kept:** selection relies on the verified invariant that terminal nodes are leaves, avoiding terminal checks along already-expanded paths.
- **Kept:** one reused ThreadRng handle now feeds compact rollout/child sampling; SmallRng remains rejected because changed trajectories regressed midgames.
- **Kept:** internal cached UCB selection scans the verified contiguous arena child slice directly and scores adjacent nodes in pairs, exposing prefetch and memory-level parallelism. Wider four-way/tournament specializations regressed.
- **Rejected:** exact child `Vec` reserve, arena reserve-by-iterations, reusable expansion action buffer, scalar UCB algebra, final-q one-pass scan, eager inlining, q-update branches, and bounds-check removal. None improved the primary metric reliably.
- **Rejected architectural prototype:** a compact hot-node sidecar improved the 512-deep chain ~9% but regressed empty TTT ~10%, duplicated public stats, and made public fields stale until synchronization. A true breaking hot/cold redesign remains promising.
- **Kept:** rollout-specific UTTT transitions check only the moving player's newly changed mini-board, and a compatible terminal-reward hook fuses terminal detection with final reward checks in both bundled games.
- **Kept:** expansion Nodes are initialized directly in reserved arena spare capacity and opted-in bundled states are then mutated in final storage. This removes intermediate 64-104 byte Node/state copies; a default-false State capability preserves the original `step` and clone behavior for external states. The small unsafe initialization is guarded by reserve-before-write and set-length-after-full-initialization invariants.
- The current six-core-workload primary best is `geomean_ns=895365.783` on the nine-round CPU-6 harness, 24.7% below the core-search baseline. Before the metric reset, the eight-workload score was 71.3% below baseline. Latest Criterion cross-check after direct construction: 10k TTT searches are 80-83% faster and 1k UTTT searches 84-85% faster versus untouched `initial`; current full-game medians are 0.372 ms for TTT n500 and 9.19 ms for UTTT n250.
- **Kept child layout:** immutable child lists use a 16-byte slice representation, store zero/one child inline, and allocate only for 2+ children. Current node sizes are TTT 64 B, UTTT 104 B, Chain 72 B, Wide 88 B (all 8 B smaller than the compact-state Vec layout); Valgrind memcheck found no errors or leaks.
