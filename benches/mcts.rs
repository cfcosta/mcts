//! Benchmark suite covering every hot path of the search separately, so a
//! change's effect can be attributed to the code path it touches:
//!
//! - `search/*`: end-to-end `Mcts::new` + `search`, the number users feel.
//! - `full_game/*`: a whole self-play game (repeated tree construction).
//! - `tree_ops/*`: synthetic games that make the state nearly free, so
//!   selection/backpropagation (deep chain) and expansion/UCB scanning
//!   (wide game) dominate.
//! - `state/*`: the `State` trait implementations in isolation (clone, step,
//!   legal-action generation, terminal check) — the per-node costs the search
//!   pays thousands of times per call.
//! - `rollout/*`: a random playout loop, mirroring the simulation phase.
//!
//! Run with `cargo bench`, filter with e.g. `cargo bench -- state/`, and
//! compare against a saved baseline as described in BENCHMARKING.md.

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mcts_rs::{Bump, Mcts, State};
use rand::seq::SliceRandom;

#[path = "../tests/support/mod.rs"]
mod support;
use support::*;

/// The exploration constants the examples use.
const TTT_C: f64 = 0.5;
const UTTT_C: f64 = 1.4142356237;

fn ttt_positions() -> [(&'static str, TicTacToe); 2] {
    [
        ("empty", ttt_after(&[])),
        ("midgame", ttt_after(&[(1, 1), (0, 0), (2, 0), (0, 2)])),
    ]
}

fn uttt_positions() -> [(&'static str, UltimateTicTacToe); 2] {
    [
        ("empty", uttt_after_plies(0)),
        ("midgame", uttt_after_plies(6)),
    ]
}

fn bench_search_tic_tac_toe(c: &mut Criterion) {
    let mut group = c.benchmark_group("search/tic_tac_toe");
    group.measurement_time(Duration::from_secs(10));
    for (name, position) in &ttt_positions() {
        for n in [100usize, 1_000, 10_000] {
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new(*name, n), &n, |b, &n| {
                let mut bump = Bump::new();
                b.iter(|| {
                    let action =
                        black_box(Mcts::new(&bump, black_box(position.clone()), TTT_C).search(n));
                    bump.reset();
                    action
                });
            });
        }
    }
    group.finish();
}

fn bench_search_ultimate(c: &mut Criterion) {
    let mut group = c.benchmark_group("search/ultimate");
    group.sample_size(30);
    group.measurement_time(Duration::from_secs(12));
    for (name, position) in &uttt_positions() {
        for n in [100usize, 1_000] {
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new(*name, n), &n, |b, &n| {
                let mut bump = Bump::new();
                b.iter(|| {
                    let action =
                        black_box(Mcts::new(&bump, black_box(position.clone()), UTTT_C).search(n));
                    bump.reset();
                    action
                });
            });
        }
    }
    group.finish();
}

fn bench_full_game(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_game");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));
    group.bench_function("tic_tac_toe_n500", |b| {
        let mut bump = Bump::new();
        b.iter(|| {
            let mut state = TicTacToe::new();
            while !state.is_terminal() {
                let action = Mcts::new(&bump, state.clone(), TTT_C).search(500);
                bump.reset();
                state = state.step(action);
            }
            black_box(state.player_has_won(0))
        })
    });
    group.bench_function("ultimate_n250", |b| {
        let mut bump = Bump::new();
        b.iter(|| {
            let mut state = UltimateTicTacToe::new();
            while !state.is_terminal() {
                let action = Mcts::new(&bump, state.clone(), UTTT_C).search(250);
                bump.reset();
                state = state.step(action);
            }
            black_box(state.player_has_won(0))
        })
    });
    group.finish();
}

fn bench_tree_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_ops");
    group.measurement_time(Duration::from_secs(10));

    // Deep, narrow tree: after the first 512 iterations every iteration
    // walks the full chain down and back up, so selection and
    // backpropagation dominate.
    let n = 2_000usize;
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("deep_chain_select_backprop", |b| {
        let mut bump = Bump::new();
        b.iter(|| {
            let action = Mcts::new(&bump, ChainGame::new(512), 1.0).search(n);
            bump.reset();
            action
        });
    });

    // Wide, shallow tree: one expansion of `width` children, then UCB scans
    // across all siblings on every iteration.
    for width in [64usize, 512] {
        let n = 2 * width;
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("wide_expand_ucb", width),
            &width,
            |b, &width| {
                let mut bump = Bump::new();
                b.iter(|| {
                    let action = black_box(Mcts::new(&bump, WideGame::new(width), 1.0).search(n));
                    bump.reset();
                    action
                });
            },
        );
    }
    group.finish();
}

fn bench_state_tic_tac_toe(c: &mut Criterion) {
    let mut group = c.benchmark_group("state/tic_tac_toe");
    let mid = ttt_after(&[(1, 1), (0, 0), (2, 0), (0, 2)]);
    let action = mid.get_legal_actions()[0];
    group.bench_function("clone", |b| b.iter(|| black_box(&mid).clone()));
    group.bench_function("step", |b| {
        b.iter(|| black_box(&mid).step(black_box(action)))
    });
    group.bench_function("legal_actions", |b| {
        b.iter(|| black_box(&mid).get_legal_actions())
    });
    group.bench_function("is_terminal", |b| b.iter(|| black_box(&mid).is_terminal()));
    group.finish();
}

fn bench_state_ultimate(c: &mut Criterion) {
    let mut group = c.benchmark_group("state/ultimate");
    let mid = uttt_after_plies(6);
    let action = mid.get_legal_actions()[0];
    group.bench_function("clone", |b| b.iter(|| black_box(&mid).clone()));
    group.bench_function("step", |b| {
        b.iter(|| black_box(&mid).step(black_box(action)))
    });
    group.bench_function("legal_actions", |b| {
        b.iter(|| black_box(&mid).get_legal_actions())
    });
    group.bench_function("is_terminal", |b| b.iter(|| black_box(&mid).is_terminal()));
    group.finish();
}

/// Mirrors the simulation phase of the search: clone the state, then play
/// uniformly random moves until the game ends.
fn random_playout<S: State + Clone>(start: &S) -> f32 {
    let mut state = start.clone();
    while !state.is_terminal() {
        let actions = state.get_legal_actions();
        let action = *actions.choose(&mut rand::thread_rng()).unwrap();
        state = state.step(action);
    }
    state.reward(0)
}

fn bench_rollout(c: &mut Criterion) {
    let mut group = c.benchmark_group("rollout");
    group.measurement_time(Duration::from_secs(8));
    let ttt = TicTacToe::new();
    group.bench_function("tic_tac_toe", |b| {
        b.iter(|| random_playout(black_box(&ttt)))
    });
    let uttt = UltimateTicTacToe::new();
    group.bench_function("ultimate", |b| b.iter(|| random_playout(black_box(&uttt))));
    group.finish();
}

criterion_group!(
    benches,
    bench_search_tic_tac_toe,
    bench_search_ultimate,
    bench_full_game,
    bench_tree_ops,
    bench_state_tic_tac_toe,
    bench_state_ultimate,
    bench_rollout,
);
criterion_main!(benches);
