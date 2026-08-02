//! Property-based tests for the average-strategy node-policy extension.
//!
//! Property inventory:
//!
//! - **Differential (bitwise)**: on a fresh node — zero regrets, zero
//!   strategy sums, zero solve count — one average-mode warm solve must
//!   reproduce [`solve_zero_sum_regret`] bitwise on policies, value, and
//!   exploitability: the iteration bodies are identical, the warm state
//!   starts at the cold solver's zeros, and `0.0 + x == x` makes the
//!   cumulative average equal the cold time average.
//! - **Structural**: across several average-mode solves with payoff
//!   perturbations in between, the installed policy is exactly the
//!   cumulative `strategy_sum / solve_count` (the fresh differential
//!   alone cannot distinguish cumulative from per-call averaging).
//! - **Convergence**: on matching pennies the time-average policy
//!   approaches the (1/2, 1/2) equilibrium — the regret-matching folk
//!   theorem that motivates average-strategy outputs; last iterates
//!   cycle and enjoy no such guarantee.
//! - **Structural (end-to-end)**: searches with the flag on — optionally
//!   stacked with prior-mass pruning and Dirichlet root noise — uphold
//!   every tree invariant, and a directed deep search proves the
//!   average-policy invariant holds on real non-root expanded nodes
//!   (the root's policy stays the cold equilibrium on both paths).
//! - **Config**: the extension defaults to off and a bool needs no
//!   validation envelope.

mod support;

use hegel::generators as gs;
use hegel::TestCase;
use mcts_rs::joint::{
    solve_node, solve_zero_sum_regret, Evaluation, JointSearchConfig, RootNoise, SearchOptions,
    SimultaneousTreeSearch, Tree,
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

// ---------------------------------------------------------------------------
// Solver laws.
// ---------------------------------------------------------------------------

/// Differential oracle: a fresh node's first average-mode solve IS the
/// cold solver — same zero starting state, same iteration body, and the
/// cumulative average over one call is the plain time average. Policies,
/// value, and exploitability must match bitwise.
#[hegel::test(test_cases = 256)]
fn fresh_node_average_solve_matches_the_cold_solver(tc: TestCase) {
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
        false,
    );

    solve_node(node, iterations, true, false);

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

/// Structural oracle: the installed policy is the *cumulative* average —
/// `strategy_sum / solve_count` over every solve so far, bitwise — not
/// the average of only the latest call, even as payoff updates between
/// calls move the underlying iterates.
#[hegel::test(test_cases = 128)]
fn average_solves_install_the_cumulative_average(tc: TestCase) {
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
        solve_node(node, batch, true, false);
    }

    let node = tree.node(id);
    assert_eq!(node.solve_count, batches.iter().sum::<u32>());
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
        for action in 0..n {
            if legal.contains(&action) {
                assert_eq!(
                    policy[action].to_bits(),
                    (sums[action] / f64::from(node.solve_count)).to_bits(),
                    "{side} action {action} carries the cumulative average"
                );
            } else {
                assert_eq!(
                    policy[action], 0.0,
                    "{side} action {action} stays off-legal"
                );
            }
        }
    }
}

/// Convergence anchor: regret matching's *average* strategy approaches
/// the mixed equilibrium of matching pennies from skewed priors; the
/// last iterate cycles around it with no such guarantee. 2048 total
/// iterations bring the O(1/sqrt(T)) average error well inside 0.05.
#[test]
fn average_policies_converge_on_matching_pennies() {
    let config = JointSearchConfig::default();
    let mut tree: Tree<ToySnapshot> = Tree::new(2);
    let id = tree.make_node(
        ToySnapshot::live(1, 0b11, 0b11),
        Evaluation {
            player_priors: vec![0.9, 0.1],
            enemy_priors: vec![0.2, 0.8],
            value: 0.0,
        },
        &config,
    );
    let node = tree.node_mut(id);
    node.record_value(0, 0, 1.0);
    node.record_value(0, 1, -1.0);
    node.record_value(1, 0, -1.0);
    node.record_value(1, 1, 1.0);
    for _ in 0..128 {
        solve_node(node, 16, true, false);
    }

    let node = tree.node(id);
    for (side, policy) in [
        ("player", &node.player_policy),
        ("enemy", &node.enemy_policy),
    ] {
        for (action, &mass) in policy.iter().enumerate() {
            assert!(
                (mass - 0.5).abs() <= 0.05,
                "{side} action {action}: average {mass} missed the 1/2 equilibrium"
            );
        }
    }
    assert!(
        node.root_value.abs() <= 0.05,
        "matching pennies is worth 0, got {}",
        node.root_value
    );
}

// ---------------------------------------------------------------------------
// End-to-end searches.
// ---------------------------------------------------------------------------

/// Directed depth check: a deep search with the flag on expands nodes
/// below the root and installs the cumulative average on every one of
/// them. Keeps the sweep's average-policy invariant from passing
/// vacuously on trees whose only expanded node is the root (whose
/// policy the cold root equilibrium overwrites on both paths).
#[test]
fn deep_average_search_installs_average_policies_below_the_root() {
    let config = JointSearchConfig {
        average_strategy_policies: true,
        max_depth: 2,
        expansion_budget: 16,
        regret_iterations: 64,
        ..JointSearchConfig::default()
    };
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
    let mut search = SimultaneousTreeSearch::new(config.clone(), 11);
    let (result, tree) = search.search_with_tree(
        &mut provider,
        &mut evaluator,
        TwoStage::root(),
        SearchOptions::default(),
    );

    assert!(result.failure.is_none(), "unexpected divergence");
    assert_joint_tree_invariants(&tree, &result, &config, "deep average");

    let mut deep_expanded = 0usize;
    for (index, node) in tree.nodes.iter().enumerate().skip(1) {
        if !node.expanded {
            continue;
        }
        deep_expanded += 1;
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
            for &action in legal {
                assert_eq!(
                    policy[action].to_bits(),
                    (sums[action] / f64::from(node.solve_count)).to_bits(),
                    "node {index} {side} action {action} carries the cumulative average"
                );
            }
        }
    }
    assert!(
        deep_expanded > 0,
        "the deep search must expand at least one non-root node"
    );
}

/// Structural oracle: random average-strategy searches — optionally
/// stacked with prior-mass pruning and Dirichlet root noise — uphold
/// every tree invariant, including the checker's average-policy law on
/// non-root expanded nodes.
#[hegel::test]
fn average_strategy_searches_uphold_every_tree_invariant(tc: TestCase) {
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
        average_strategy_policies: true,
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
    assert_joint_tree_invariants(&tree, &result, &config, "average-strategy scenario");
}

// ---------------------------------------------------------------------------
// Configuration surface.
// ---------------------------------------------------------------------------

/// The extension defaults to off — the defaults keep the last-iterate
/// behavior — and a bool has no validation envelope to reject.
#[test]
fn average_strategy_defaults_off_and_validates() {
    let config = JointSearchConfig::default();
    assert!(!config.average_strategy_policies);
    assert_eq!(config.validate(), Ok(()));

    let enabled = JointSearchConfig {
        average_strategy_policies: true,
        ..JointSearchConfig::default()
    };
    assert_eq!(enabled.validate(), Ok(()));
}
