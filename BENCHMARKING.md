# Benchmarking & correctness workflow

This repo has two safety nets for performance work: a **behavioral test
suite** that pins what the search must keep doing, and a **Criterion
benchmark suite** that measures each hot path separately. The intended loop:

```sh
# 1. Record a baseline on the unmodified code
cargo bench -- --save-baseline before

# 2. Make your change, then check nothing broke
cargo test

# 3. Compare performance against the baseline
cargo bench -- --baseline before
```

Criterion prints a per-benchmark comparison (improved / regressed / no
change) and writes HTML reports with plots to `target/criterion/report/index.html`.
To benchmark only one code path while iterating, filter by group name, e.g.
`cargo bench -- state/` or `cargo bench -- tree_ops`.

## What the benchmarks measure

Each group isolates a different code path, so a change's effect can be
attributed:

| Group | What dominates | What it tells you |
|---|---|---|
| `search/tic_tac_toe` | end-to-end search (n = 100 / 1k / 10k, empty & midgame) | the number users feel; per-iteration throughput |
| `search/ultimate` | end-to-end search on a big branching factor | same, with expensive states and long playouts |
| `full_game/*` | whole self-play games | repeated tree construction + search across a game |
| `tree_ops/deep_chain_select_backprop` | selection + backpropagation on a 512-deep chain with a near-free state | tree-walking machinery in isolation |
| `tree_ops/wide_expand_ucb` | expansion + UCB scans over 64 / 512 siblings | node creation and best-child selection |
| `state/tic_tac_toe`, `state/ultimate` | `clone` / `step` / `get_legal_actions` / `is_terminal` in isolation | the per-node costs the search pays thousands of times per call (currently dominated by `Vec` allocation) |
| `rollout/*` | a random playout loop | the simulation phase |

Search benchmarks report throughput in elements/sec where one element = one
search iteration. The search is stochastic (random playouts), so per-sample
times vary more than in a deterministic benchmark; Criterion's sampling
averages this out, but prefer a quiet machine and treat < ~3% deltas as noise.

## What the tests guarantee

`cargo test` runs in a few seconds and must stay green through any
performance change. The suite checks the library from the outside in:

- **Exact semantics** (`tests/deterministic.rs`): on games built so random
  playouts cannot affect the outcome, the tree shape, visit counts, q values,
  and chosen action are asserted *exactly* — one node expanded per iteration,
  chain-shaped trees for single-action games, exact +1/0/−1 arm values for a
  bandit. Any failure here is a real behavior change, never a flake.
- **Tree invariants** (`tests/tree_invariants.rs`, checked across games ×
  positions × exploration constants × iteration counts): root visits equal
  iterations, parent/child links are mutually consistent, the arena contains
  exactly the reachable tree, `q == reward_sum / n`, unvisited nodes are
  untouched leaves, expansion is all-or-nothing (one child per legal action),
  terminal nodes are never expanded, and every iteration passes through
  exactly one root child.
- **Playing strength** (`tests/behavior_tic_tac_toe.rs`,
  `tests/behavior_ultimate.rs`): the search always returns legal actions,
  takes an immediate win, blocks an immediate loss, prefers winning to
  blocking, essentially always draws Tic-Tac-Toe self-play, and does not lose
  to a random opponent from either side.

### A note on flakiness

The search uses unseeded random playouts, so the playing-strength tests are
statistical. Their thresholds were calibrated by measuring failure rates over
400–800 games per scenario: tactical positions missed 0/2000 trials, and the
self-play/vs-random tests allow the small measured tail (e.g. ~0.5% of
self-play games are decisive at n = 2000 — inherent to the current
implementation, which picks the final move by max q) with thresholds that
put the false-failure probability around 10⁻⁴ while still failing with near
certainty under any real play-quality regression. **Treat a failure as a
regression, not a flake**; if you must re-run, twice-in-a-row failure is
conclusive.

Two current edge-case behaviors are pinned by `#[should_panic]` tests:
`search(0)` and searching from a terminal state both panic (no children to
choose from). If you deliberately change that contract, update those tests.

## Repo layout for this workflow

- `examples/` — the canonical `TicTacToe` / `UltimateTicTacToe`
  implementations, each a self-contained demo of the `State` trait.
- `tests/support/games/` — the copy of those games shared by the tests and
  benches (a third copy lives in `.auto/bench/src/games/`). The library
  itself no longer ships any game; keep the copies in sync by hand when a
  game changes.
- `tests/support/mod.rs` — shared test/bench helpers: deterministic synthetic
  games (`ChainGame`, `BanditGame`, `WideGame`), seeded-opponent game
  drivers, and the tree-invariant checker.
- `benches/mcts.rs` — the Criterion suite (`[[bench]] name = "mcts"`).

Unspecified behavior you may change freely: which action is returned when
several children tie exactly on q, iteration order over equal candidates,
and anything about `render`. Everything else observable is covered above.
