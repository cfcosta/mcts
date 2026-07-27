# Autoresearch: MCTS throughput and cache locality

## Objective
Improve the real performance of the generic MCTS implementation and bundled game states. Focus on vectorization opportunities, cache-local data layouts, fewer allocations and pointer chases, and less work in hot loops while preserving all observable behavior.

## Metrics
- **Primary**: `geomean_ns` (ns, lower is better) — geometric mean of eight representative Criterion medians so each workload contributes by relative speedup rather than absolute duration.
- **Secondary**: `ttt_empty_ns`, `ttt_mid_ns`, `uttt_empty_ns`, `uttt_mid_ns`, `deep_chain_ns`, `wide_512_ns`, `uttt_rollout_ns`, `uttt_step_ns`.

The primary workload covers empty and midgame end-to-end searches for both bundled games, deep selection/backpropagation, wide expansion/UCB scans, Ultimate Tic-Tac-Toe rollout, and state stepping. Periodically run broader benchmark groups when a structural change may expose a tradeoff.

## How to Run
`./.auto/measure.sh` — enters a Nix shell providing `cargo` and `rustc`, runs the fixed-work harness in `.auto/bench/` pinned to one CPU, and emits `METRIC name=value` lines. The harness mirrors the corresponding cases in `benches/mcts.rs` but uses five batched medians because hard-coded Criterion measurement durations made an autonomous iteration take over 80 seconds.

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
- **Kept:** optional allocation-free uniformly random action selection for rollouts. UTTT uses board popcounts plus nth-set-bit selection; this was confirmed twice and improved UTTT searches ~22-23% after earlier direct empty-cell selection had already removed rollout vectors.
- **Kept:** 512-byte compile-time win lookup tables for both games and empty-bit iteration for UTTT legal-action generation.
- **Rejected:** reusing one ThreadRng handle, exact child `Vec` reserve, arena reserve-by-iterations, reusable expansion action buffer, scalar UCB algebra, direct contiguous-child slicing, final-q one-pass scan, and TTT set-bit action iteration. None improved the primary metric reliably.
- **Rejected architectural prototype:** a compact hot-node sidecar improved the 512-deep chain ~9% but regressed empty TTT ~10%, duplicated public stats, and made public fields stale until synchronization. A true breaking hot/cold redesign remains promising.
- Current fixed-work best is `geomean_ns=205184.905`, 52.9% below the fixed-work baseline.
