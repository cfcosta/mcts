//! Property-based tests for the CFR+-style solve extension: alternating
//! regret updates with linearly weighted strategy averaging.
//!
//! Property inventory:
//!
//! - **Algebraic (bitwise)**: `strategy_weight_total` is the plain count
//!   under the uniform scheme and the exact triangular number `S(S+1)/2`
//!   under CFR+ — checked against an independent integer oracle. Every
//!   average in the implementation and in the invariant checker
//!   normalizes through this one helper.
//! - **Differential (bitwise)**: at one iteration the variants coincide —
//!   both averages are the first iterate, whose computation precedes any
//!   update the variants disagree on. Likewise a 1x1 legal grid solves to
//!   the same exact point under either dynamic, for any iteration count.
//! - **Differential (bitwise)**: a fresh node's first CFR+ average-mode
//!   warm solve reproduces the CFR+ cold solver exactly, extending the
//!   average-strategy extension's bitwise bridge to the new dynamics.
//! - **Structural**: across batched warm solves the installed average
//!   policy is `strategy_sum / (S(S+1)/2)` bitwise, and the accumulated
//!   strategy mass tracks the triangular total — pinning the *global*
//!   iteration weights `t = solve_count + i + 1` across batches (weights
//!   restarting at 1 each batch would break the mass identity).
//! - **Convergence**: on matching pennies and rock-paper-scissors from
//!   skewed priors the weighted average reaches the mixed equilibrium,
//!   and on the fixed pennies instance CFR+ ends at least as close to
//!   equilibrium as the default simultaneous dynamics — the deterministic
//!   witness of the faster-convergence claim motivating the extension.
//! - **Structural (end-to-end)**: searches with the flag on — stacked
//!   with every other extension — uphold all tree invariants, and same
//!   seed runs are bitwise deterministic.
//! - **Config**: the extension defaults to off and a bool needs no
//!   validation envelope.

mod support;

use hegel::generators as gs;
use hegel::TestCase;
use mcts_rs::joint::{
    legal_from_priors, solve_node, solve_zero_sum_regret, strategy_weight_total, Evaluation,
    JointSearchConfig, RootNoise, SearchOptions, SimultaneousTreeSearch, Tree,
};
use support::joint::{
    assert_joint_tree_invariants, FixedPriorEvaluator, MatrixProvider, ToySnapshot, TwoStage,
};

// ---------------------------------------------------------------------------
// Draw helpers: valid inputs by construction, no rejection.
// ---------------------------------------------------------------------------

/// A non-empty legal mask over `n` actions.
fn draw_mask(tc: &TestCase, n: usize) -> u64 {
    tc.draw(gs::integers::<u64>().min_value(1).max_value((1 << n) - 1))
}

/// Priors in `[0, 1]`; zeros are allowed and exercise the uniform
/// fallback inside `normalized_prior`.
fn draw_priors(tc: &TestCase, n: usize) -> Vec<f64> {
    tc.draw(
        gs::vecs(gs::floats::<f64>().min_value(0.0).max_value(1.0))
            .min_size(n)
            .max_size(n),
    )
}

fn draw_matrix(tc: &TestCase, n: usize) -> Vec<f64> {
    tc.draw(
        gs::vecs(gs::floats::<f64>().min_value(-1.0).max_value(1.0))
            .min_size(n * n)
            .max_size(n * n),
    )
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// A fresh node over drawn masks/priors with `recorded` payoff samples
/// mixed in, ready for its first solve.
fn draw_fresh_node(tc: &TestCase, n: usize, cap: usize) -> (Tree<ToySnapshot>, u32) {
    let config = JointSearchConfig {
        max_actions_per_side: cap,
        ..JointSearchConfig::default()
    };
    let evaluation = Evaluation {
        player_priors: draw_priors(tc, n),
        enemy_priors: draw_priors(tc, n),
        value: tc.draw(gs::floats::<f64>().min_value(-1.0).max_value(1.0)),
    };
    let mut tree: Tree<ToySnapshot> = Tree::new(n);
    let id = tree.make_node(
        ToySnapshot::live(1, draw_mask(tc, n), draw_mask(tc, n)),
        evaluation,
        &config,
    );
    let node = tree.node_mut(id);
    let recorded: usize = tc.draw(gs::integers::<usize>().min_value(0).max_value(16));
    for _ in 0..recorded {
        let player: usize = tc.draw(gs::sampled_from(node.player_legal.clone()));
        let enemy: usize = tc.draw(gs::sampled_from(node.enemy_legal.clone()));
        let value: f64 = tc.draw(gs::floats::<f64>().min_value(-1.0).max_value(1.0));
        node.record_value(player, enemy, value);
    }
    (tree, id)
}

/// The matching-pennies cold-solve inputs used by the convergence tests.
fn matching_pennies() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>) {
    let payoff = vec![1.0, -1.0, -1.0, 1.0];
    let player_priors = vec![0.9, 0.1];
    let enemy_priors = vec![0.2, 0.8];
    let legal = vec![0, 1];
    (payoff, player_priors, enemy_priors, legal)
}

// ---------------------------------------------------------------------------
// The weight schedule.
// ---------------------------------------------------------------------------

/// Algebraic oracle: the uniform scheme weighs every iteration 1 (total =
/// solve count) and CFR+ weighs iteration `t` by `t` (total = the
/// triangular number), computed here independently in exact integer
/// arithmetic. Both must match bitwise — solver, checker, and search all
/// normalize through this helper.
#[hegel::test(test_cases = 128)]
fn weight_totals_are_uniform_or_triangular(tc: TestCase) {
    let solve_count: u32 = tc.draw(gs::integers::<u32>().min_value(0).max_value(100_000));
    assert_eq!(
        strategy_weight_total(false, solve_count).to_bits(),
        f64::from(solve_count).to_bits(),
        "uniform weights total the solve count"
    );
    let count = u64::from(solve_count);
    let triangular = (count * (count + 1) / 2) as f64;
    assert_eq!(
        strategy_weight_total(true, solve_count).to_bits(),
        triangular.to_bits(),
        "linear weights total the triangular number"
    );
}

// ---------------------------------------------------------------------------
// Solver laws.
// ---------------------------------------------------------------------------

/// Differential oracle: with exactly one iteration the variants must
/// coincide bitwise. Both averages equal the first iterate, which is
/// computed from the zero-initialized regrets before either variant's
/// update path can diverge, and the weight of iteration one is 1 under
/// both schedules.
#[hegel::test(test_cases = 256)]
fn one_iteration_solves_coincide_across_variants(tc: TestCase) {
    let n: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(5));
    let cap: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(5));
    let payoff = draw_matrix(&tc, n);
    let player_priors = draw_priors(&tc, n);
    let enemy_priors = draw_priors(&tc, n);
    let player_legal = legal_from_priors(draw_mask(&tc, n), &player_priors, cap);
    let enemy_legal = legal_from_priors(draw_mask(&tc, n), &enemy_priors, cap);

    let uniform = solve_zero_sum_regret(
        &payoff,
        n,
        &player_priors,
        &enemy_priors,
        &player_legal,
        &enemy_legal,
        1,
        false,
    );
    let cfr_plus = solve_zero_sum_regret(
        &payoff,
        n,
        &player_priors,
        &enemy_priors,
        &player_legal,
        &enemy_legal,
        1,
        true,
    );

    assert_eq!(bits(&uniform.0), bits(&cfr_plus.0), "player policy");
    assert_eq!(bits(&uniform.1), bits(&cfr_plus.1), "enemy policy");
    assert_eq!(uniform.2.to_bits(), cfr_plus.2.to_bits(), "value");
    assert_eq!(uniform.3.to_bits(), cfr_plus.3.to_bits(), "exploitability");
}

/// Differential oracle: a 1x1 legal grid is a fixpoint of both dynamics
/// for any iteration count — the lone strategies are exactly 1, the value
/// is exactly the lone cell, and the gap is exactly zero — so the
/// variants must agree bitwise however long they run.
#[hegel::test(test_cases = 128)]
fn single_legal_pair_solves_are_variant_invariant(tc: TestCase) {
    let n: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
    let payoff = draw_matrix(&tc, n);
    let player_priors = draw_priors(&tc, n);
    let enemy_priors = draw_priors(&tc, n);
    let player_action: usize = tc.draw(gs::integers::<usize>().min_value(0).max_value(n - 1));
    let enemy_action: usize = tc.draw(gs::integers::<usize>().min_value(0).max_value(n - 1));
    let iterations: u32 = tc.draw(gs::integers::<u32>().min_value(1).max_value(64));

    let solve = |cfr_plus: bool| {
        solve_zero_sum_regret(
            &payoff,
            n,
            &player_priors,
            &enemy_priors,
            &[player_action],
            &[enemy_action],
            iterations,
            cfr_plus,
        )
    };
    let uniform = solve(false);
    let cfr_plus = solve(true);

    assert_eq!(bits(&uniform.0), bits(&cfr_plus.0), "player policy");
    assert_eq!(bits(&uniform.1), bits(&cfr_plus.1), "enemy policy");
    assert_eq!(uniform.2.to_bits(), cfr_plus.2.to_bits(), "value");
    assert_eq!(uniform.3.to_bits(), cfr_plus.3.to_bits(), "exploitability");
    assert_eq!(cfr_plus.0[player_action], 1.0, "lone player action");
    assert_eq!(cfr_plus.1[enemy_action], 1.0, "lone enemy action");
    let cell = payoff[player_action * n + enemy_action];
    assert_eq!(cfr_plus.2.to_bits(), cell.to_bits(), "value is the cell");
    assert_eq!(cfr_plus.3, 0.0, "a 1x1 grid has no gap");
}

/// Differential oracle: a fresh node's first CFR+ average-mode solve IS
/// the CFR+ cold solver — same zero starting state, same alternating
/// iteration body, and the batch-local weights `0 + i + 1` equal the
/// cold solver's `i + 1` exactly. Policies, value, and exploitability
/// must match bitwise.
#[hegel::test(test_cases = 256)]
fn fresh_node_cfr_plus_average_solve_matches_the_cold_solver(tc: TestCase) {
    let n: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(5));
    let cap: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(5));
    let iterations: u32 = tc.draw(gs::integers::<u32>().min_value(1).max_value(64));
    let (mut tree, id) = draw_fresh_node(&tc, n, cap);

    let node = tree.node_mut(id);
    let (cold_player, cold_enemy, cold_value, cold_gap) = solve_zero_sum_regret(
        &node.payoff.clone(),
        n,
        &node.player_priors.clone(),
        &node.enemy_priors.clone(),
        &node.player_legal.clone(),
        &node.enemy_legal.clone(),
        iterations,
        true,
    );

    solve_node(node, iterations, true, true);

    let node = tree.node(id);
    assert_eq!(node.solve_count, iterations);
    assert_eq!(
        bits(&node.player_policy),
        bits(&cold_player),
        "player policy"
    );
    assert_eq!(bits(&node.enemy_policy), bits(&cold_enemy), "enemy policy");
    assert_eq!(node.root_value.to_bits(), cold_value.to_bits(), "value");
    assert_eq!(
        node.exploitability.to_bits(),
        cold_gap.to_bits(),
        "exploitability"
    );
}

/// Structural oracle: across batched CFR+ warm solves the installed
/// policy is the linearly weighted cumulative average — `strategy_sum`
/// over the triangular weight total, bitwise — and the accumulated mass
/// tracks that total, which only holds when the iteration weights
/// continue globally across batches (`t = solve_count + i + 1`).
#[hegel::test(test_cases = 128)]
fn cfr_plus_average_solves_install_the_cumulative_weighted_average(tc: TestCase) {
    let n: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(5));
    let cap: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(5));
    let batches: Vec<u32> = tc.draw(
        gs::vecs(gs::integers::<u32>().min_value(1).max_value(32))
            .min_size(1)
            .max_size(4),
    );
    let (mut tree, id) = draw_fresh_node(&tc, n, cap);

    let node = tree.node_mut(id);
    for &batch in &batches {
        let player: usize = tc.draw(gs::sampled_from(node.player_legal.clone()));
        let enemy: usize = tc.draw(gs::sampled_from(node.enemy_legal.clone()));
        let value: f64 = tc.draw(gs::floats::<f64>().min_value(-1.0).max_value(1.0));
        node.record_value(player, enemy, value);
        solve_node(node, batch, true, true);
    }

    let node = tree.node(id);
    assert_eq!(node.solve_count, batches.iter().sum::<u32>());
    let total = strategy_weight_total(true, node.solve_count);
    for (side, legal, sums, policy) in [
        (
            "player",
            &node.player_legal,
            &node.player_strategy_sum,
            &node.player_policy,
        ),
        (
            "enemy",
            &node.enemy_legal,
            &node.enemy_strategy_sum,
            &node.enemy_policy,
        ),
    ] {
        let mut mass = 0.0;
        for action in 0..n {
            if legal.contains(&action) {
                assert_eq!(
                    policy[action].to_bits(),
                    (sums[action] / total).to_bits(),
                    "{side} action {action} carries the weighted cumulative average"
                );
                mass += sums[action];
            } else {
                assert_eq!(
                    policy[action], 0.0,
                    "{side} action {action} stays off-legal"
                );
            }
        }
        assert!(
            (mass - total).abs() <= 1e-9 * total.max(1.0),
            "{side} strategy mass {mass} must track the triangular total {total}"
        );
    }
}

/// Convergence anchor and the extension's deterministic optimality
/// witness: from skewed priors on matching pennies, 2048 CFR+ iterations
/// land the weighted average inside 0.05 of the (1/2, 1/2) equilibrium,
/// and no farther from it than the default simultaneous uniform average —
/// the faster-convergence claim, checked on a fixed instance.
#[test]
fn cfr_plus_converges_on_matching_pennies_at_least_as_fast() {
    let (payoff, player_priors, enemy_priors, legal) = matching_pennies();
    let solve = |cfr_plus: bool| {
        solve_zero_sum_regret(
            &payoff,
            2,
            &player_priors,
            &enemy_priors,
            &legal,
            &legal,
            2048,
            cfr_plus,
        )
    };
    let uniform = solve(false);
    let cfr_plus = solve(true);

    for (side, policy) in [("player", &cfr_plus.0), ("enemy", &cfr_plus.1)] {
        for (action, &mass) in policy.iter().enumerate() {
            assert!(
                (mass - 0.5).abs() <= 0.05,
                "{side} action {action}: average {mass} missed the 1/2 equilibrium"
            );
        }
    }
    assert!(
        cfr_plus.2.abs() <= 0.05,
        "matching pennies is worth 0, got {}",
        cfr_plus.2
    );
    assert!(
        cfr_plus.3 >= -1e-9,
        "exploitability stays non-negative, got {}",
        cfr_plus.3
    );
    assert!(
        cfr_plus.3 <= uniform.3,
        "CFR+ gap {} must not exceed the uniform-average gap {}",
        cfr_plus.3,
        uniform.3
    );
}

/// Convergence anchor on a 3x3 cycle: rock-paper-scissors from skewed
/// priors has the unique mixed equilibrium (1/3, 1/3, 1/3), which the
/// weighted average must approach — alternation bugs that let one side
/// respond to a stale strategy show up as asymmetric drift here.
#[test]
fn cfr_plus_converges_on_rock_paper_scissors() {
    let payoff = vec![0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0];
    let (player, enemy, value, gap) = solve_zero_sum_regret(
        &payoff,
        3,
        &[0.6, 0.3, 0.1],
        &[0.2, 0.3, 0.5],
        &[0, 1, 2],
        &[0, 1, 2],
        4096,
        true,
    );

    for (side, policy) in [("player", &player), ("enemy", &enemy)] {
        for (action, &mass) in policy.iter().enumerate() {
            assert!(
                (mass - 1.0 / 3.0).abs() <= 0.05,
                "{side} action {action}: average {mass} missed the 1/3 equilibrium"
            );
        }
    }
    assert!(value.abs() <= 0.05, "the cycle is worth 0, got {value}");
    assert!((-1e-9..=0.1).contains(&gap), "gap {gap} out of envelope");
}

// ---------------------------------------------------------------------------
// End-to-end searches.
// ---------------------------------------------------------------------------

/// Structural oracle: random CFR+ searches — stacked with the average
/// policy, prior-mass pruning, and Dirichlet noise extensions — uphold
/// every tree invariant, including the checker's weighted strategy-mass
/// and average-policy laws.
#[hegel::test]
fn cfr_plus_searches_uphold_every_tree_invariant(tc: TestCase) {
    let staged: bool = tc.draw(gs::booleans());
    let n: usize = if staged {
        2
    } else {
        tc.draw(gs::integers::<usize>().min_value(1).max_value(4))
    };
    let max_actions_per_side: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(13));
    let prune: bool = tc.draw(gs::booleans());
    let noisy: bool = tc.draw(gs::booleans());
    let config = JointSearchConfig {
        cfr_plus_solves: true,
        average_strategy_policies: tc.draw(gs::booleans()),
        root_noise: noisy.then(|| RootNoise {
            epsilon: tc.draw(gs::floats::<f64>().min_value(0.05).max_value(1.0)),
            alpha_scale: tc.draw(gs::floats::<f64>().min_value(0.05).max_value(20.0)),
        }),
        prior_mass_cutoff: prune.then(|| tc.draw(gs::floats::<f64>().min_value(0.05).max_value(1.0))),
        minimum_actions_per_side: tc.draw(
            gs::integers::<usize>()
                .min_value(1)
                .max_value(max_actions_per_side),
        ),
        max_actions_per_side,
        expansion_budget: tc.draw(gs::integers::<u32>().min_value(1).max_value(24)),
        minimum_expansion_budget: tc.draw(gs::integers::<u32>().min_value(1).max_value(24)),
        max_depth: tc.draw(gs::integers::<u32>().min_value(1).max_value(3)),
        chance_samples_per_joint: tc.draw(gs::integers::<u32>().min_value(1).max_value(2)),
        regret_iterations: tc.draw(gs::integers::<u32>().min_value(8).max_value(64)),
        regret_iterations_per_update: tc.draw(gs::integers::<u32>().min_value(1).max_value(16)),
        adaptive_search: tc.draw(gs::booleans()),
        ..JointSearchConfig::default()
    };
    let options = SearchOptions {
        sample_actions: tc.draw(gs::booleans()),
        router_score: tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0)),
    };
    let seed: u64 = tc.draw(gs::integers::<u64>());
    let mut evaluator = FixedPriorEvaluator {
        player_priors: draw_priors(&tc, n),
        enemy_priors: draw_priors(&tc, n),
        value: tc.draw(gs::floats::<f64>().min_value(-1.0).max_value(1.0)),
    };

    let mut search = SimultaneousTreeSearch::new(config.clone(), seed);
    let (result, tree) = if staged {
        let mut provider = TwoStage {
            stage_matrix: draw_matrix(&tc, 2),
            stage_potential: tc.draw(gs::floats::<f64>().min_value(-0.5).max_value(0.5)),
            bail_value: None,
        };
        search.search_with_tree(&mut provider, &mut evaluator, TwoStage::root(), options)
    } else {
        let mut provider = MatrixProvider::new(n, draw_matrix(&tc, n));
        let root = provider.root();
        search.search_with_tree(&mut provider, &mut evaluator, root, options)
    };

    assert!(result.failure.is_none(), "unexpected divergence");
    assert_joint_tree_invariants(&tree, &result, &config, "CFR+ scenario");
}

/// Bitwise determinism: two CFR+ searches from the same seed — noise,
/// averaging, and pruning all stacked on — produce identical results.
#[test]
fn cfr_plus_searches_are_bitwise_deterministic() {
    let config = JointSearchConfig {
        cfr_plus_solves: true,
        average_strategy_policies: true,
        root_noise: Some(RootNoise::default()),
        max_depth: 2,
        expansion_budget: 24,
        regret_iterations: 64,
        ..JointSearchConfig::default()
    };
    let run = || {
        let mut provider = TwoStage {
            stage_matrix: vec![1.0, -1.0, -1.0, 1.0],
            stage_potential: 0.25,
            bail_value: None,
        };
        let mut evaluator = FixedPriorEvaluator {
            player_priors: vec![0.7, 0.3],
            enemy_priors: vec![0.6, 0.4],
            value: 0.1,
        };
        let mut search = SimultaneousTreeSearch::new(config.clone(), 17);
        search.search(
            &mut provider,
            &mut evaluator,
            TwoStage::root(),
            SearchOptions::default(),
        )
    };
    assert_eq!(run(), run(), "same seed, same CFR+ search");
}

// ---------------------------------------------------------------------------
// Configuration surface.
// ---------------------------------------------------------------------------

/// The extension defaults to off — the defaults keep the uniform-average
/// simultaneous dynamics — and a bool has no validation envelope to
/// reject.
#[test]
fn cfr_plus_defaults_off_and_validates() {
    let config = JointSearchConfig::default();
    assert!(!config.cfr_plus_solves);
    assert_eq!(config.validate(), Ok(()));

    let enabled = JointSearchConfig {
        cfr_plus_solves: true,
        average_strategy_policies: true,
        ..JointSearchConfig::default()
    };
    assert_eq!(enabled.validate(), Ok(()));
}
