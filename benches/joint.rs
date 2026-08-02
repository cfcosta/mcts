//! Benchmarks for the joint simultaneous-move search, one per hot path:
//!
//! - `joint/cold_solve_13x13_2048`: the root equilibrium — the dominant
//!   fixed cost of every search call.
//! - `joint/warm_solve_16_13x13`: the incremental node solve paid once
//!   per learned simulation.
//! - `joint/root_only_13`: an end-to-end root-only search at the default
//!   action cap of 13 per side (169-cell install plus equilibrium).
//! - `joint/deep_two_stage_budget_320`: an end-to-end deep search at the
//!   default transition budget, exercising descent, resampling, and the
//!   convergence machinery.
//! - `joint/deep_two_stage_tolerance`: the same deep search with the
//!   opt-in `equilibrium_tolerance`, letting the two cold equilibria
//!   stop at their first converged checkpoint.
//! - `joint/deep_chain_256`: an end-to-end search down an endless
//!   two-action chain with a 256-ply depth horizon. Every level shares
//!   one child, so the cost is pure descent depth: ~255 progressively
//!   deeper simulations, each solving every node it touches. The
//!   workload is pinned by `deep_chain_bench_workload_saturates_its_
//!   256_level_horizon` in `tests/joint_behavior.rs`.
//!
//! Run with `cargo bench --bench joint`.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use mcts_rs::joint::{
    rng::next_f64, solve_node, solve_zero_sum_regret, Evaluation, JointSearchConfig, SearchOptions,
    SimultaneousTreeSearch, SplitMix64, Tree,
};

#[path = "../tests/support/mod.rs"]
mod support;
use support::joint::{DeepChain, MatrixProvider, ToySnapshot, TwoStage, UniformEvaluator};

/// A deterministic payoff matrix with entries in [-1, 1].
fn pseudo_matrix(action_count: usize, seed: u64) -> Vec<f64> {
    let mut rng = SplitMix64::new(seed);
    (0..action_count * action_count)
        .map(|_| next_f64(&mut rng) * 2.0 - 1.0)
        .collect()
}

/// Deterministic strictly positive unnormalized priors.
fn pseudo_priors(action_count: usize, seed: u64) -> Vec<f64> {
    let mut rng = SplitMix64::new(seed);
    (0..action_count)
        .map(|_| next_f64(&mut rng) + 0.05)
        .collect()
}

fn bench_cold_solve(c: &mut Criterion) {
    let n = 13;
    let payoff = pseudo_matrix(n, 1);
    let player_priors = pseudo_priors(n, 2);
    let enemy_priors = pseudo_priors(n, 3);
    let legal: Vec<usize> = (0..n).collect();
    c.bench_function("joint/cold_solve_13x13_2048", |b| {
        b.iter(|| {
            black_box(solve_zero_sum_regret(
                black_box(&payoff),
                n,
                &player_priors,
                &enemy_priors,
                &legal,
                &legal,
                2048,
                false,
            ))
        });
    });
}

fn bench_warm_solve(c: &mut Criterion) {
    let n = 13;
    let mask = (1u64 << n) - 1;
    let config = JointSearchConfig::default();
    let mut tree: Tree<ToySnapshot> = Tree::new(n);
    let node_id = tree.make_node(
        ToySnapshot::live(0, mask, mask),
        Evaluation {
            player_priors: pseudo_priors(n, 2),
            enemy_priors: pseudo_priors(n, 3),
            value: 0.0,
        },
        &config,
    );
    let node = tree.node_mut(node_id);
    let values = pseudo_matrix(n, 4);
    for player in 0..n {
        for enemy in 0..n {
            node.record_value(player, enemy, values[player * n + enemy]);
        }
    }
    // Warm solves accumulate on the node between iterations, exactly as
    // repeated learned simulations do on a live root.
    c.bench_function("joint/warm_solve_16_13x13", |b| {
        b.iter(|| solve_node(black_box(&mut *node), 16, false, false));
    });
}

fn bench_root_only(c: &mut Criterion) {
    let n = 13;
    let matrix = pseudo_matrix(n, 5);
    let config = JointSearchConfig {
        expansion_budget: 1,
        minimum_expansion_budget: 1,
        ..JointSearchConfig::default()
    };
    c.bench_function("joint/root_only_13", |b| {
        b.iter(|| {
            let mut provider = MatrixProvider::new(n, matrix.clone());
            let mut evaluator = UniformEvaluator {
                action_count: n,
                value: 0.0,
            };
            let mut search = SimultaneousTreeSearch::new(config.clone(), 17);
            let root = provider.root();
            black_box(search.search(
                &mut provider,
                &mut evaluator,
                root,
                SearchOptions::default(),
            ))
        });
    });
}

fn bench_deep_two_stage(c: &mut Criterion) {
    let config = JointSearchConfig {
        max_depth: 3,
        ..JointSearchConfig::default()
    };
    c.bench_function("joint/deep_two_stage_budget_320", |b| {
        b.iter(|| {
            let mut provider = TwoStage {
                stage_matrix: vec![1.0, -1.0, -1.0, 1.0],
                stage_potential: 0.15,
                bail_value: Some(-1.0),
            };
            let mut evaluator = UniformEvaluator {
                action_count: 2,
                value: -0.2,
            };
            let mut search = SimultaneousTreeSearch::new(config.clone(), 23);
            black_box(search.search(
                &mut provider,
                &mut evaluator,
                TwoStage::root(),
                SearchOptions::default(),
            ))
        });
    });
}

fn bench_deep_two_stage_tolerance(c: &mut Criterion) {
    let config = JointSearchConfig {
        max_depth: 3,
        equilibrium_tolerance: Some(0.005),
        ..JointSearchConfig::default()
    };
    c.bench_function("joint/deep_two_stage_tolerance", |b| {
        b.iter(|| {
            let mut provider = TwoStage {
                stage_matrix: vec![1.0, -1.0, -1.0, 1.0],
                stage_potential: 0.15,
                bail_value: Some(-1.0),
            };
            let mut evaluator = UniformEvaluator {
                action_count: 2,
                value: -0.2,
            };
            let mut search = SimultaneousTreeSearch::new(config.clone(), 23);
            black_box(search.search(
                &mut provider,
                &mut evaluator,
                TwoStage::root(),
                SearchOptions::default(),
            ))
        });
    });
}

fn bench_deep_chain(c: &mut Criterion) {
    let config = JointSearchConfig {
        max_depth: 256,
        expansion_budget: 16_384,
        minimum_expansion_budget: 1,
        ..JointSearchConfig::default()
    };
    let mut group = c.benchmark_group("joint");
    // Each iteration is a full ~255-simulation search, far heavier than
    // the other benches; fewer samples keep the run time reasonable.
    group.sample_size(10);
    group.bench_function("deep_chain_256", |b| {
        b.iter(|| {
            let mut provider = DeepChain { action_count: 2 };
            let mut evaluator = UniformEvaluator {
                action_count: 2,
                value: 0.0,
            };
            let mut search = SimultaneousTreeSearch::new(config.clone(), 29);
            let root = provider.root();
            black_box(search.search(
                &mut provider,
                &mut evaluator,
                root,
                SearchOptions::default(),
            ))
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_cold_solve,
    bench_warm_solve,
    bench_root_only,
    bench_deep_two_stage,
    bench_deep_two_stage_tolerance,
    bench_deep_chain
);
criterion_main!(benches);
