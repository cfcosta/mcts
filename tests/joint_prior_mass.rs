//! Property tests for opt-in prior-mass action pruning: when
//! `prior_mass_cutoff` is set, each side's legal list shrinks to the
//! smallest high-prior prefix holding that share of the raw prior mass,
//! never below `minimum_actions_per_side` actions.
//!
//! Oracles:
//!
//! - **Differential**: the kept count is checked against an independent
//!   naive reference over the gathered prior values.
//! - **Metamorphic**: a cutoff of `1.0` over strictly positive priors is
//!   a no-op on the list, and the end-to-end search is then bitwise
//!   identical to a search with pruning disabled.
//! - **Structural**: random pruned searches uphold every check in
//!   `assert_joint_tree_invariants`, whose legal-list reconstruction
//!   applies the same truncation rule.

mod support;

use hegel::generators as gs;
use hegel::TestCase;
use mcts_rs::joint::{
    legal_from_priors, truncate_to_prior_mass, JointSearchConfig, SearchOptions,
    SimultaneousTreeSearch,
};
use support::joint::{assert_joint_tree_invariants, FixedPriorEvaluator, MatrixProvider, TwoStage};

// ---------------------------------------------------------------------------
// Draw helpers: valid inputs by construction, no rejection.
// ---------------------------------------------------------------------------

fn draw_mask(tc: &TestCase, n: usize) -> u64 {
    tc.draw(gs::integers::<u64>().min_value(1).max_value((1 << n) - 1))
}

fn draw_priors(tc: &TestCase, n: usize) -> Vec<f64> {
    tc.draw(
        gs::vecs(gs::floats::<f64>().min_value(0.0).max_value(1.0))
            .min_size(n)
            .max_size(n),
    )
}

fn draw_positive_priors(tc: &TestCase, n: usize) -> Vec<f64> {
    tc.draw(
        gs::vecs(gs::floats::<f64>().min_value(0.05).max_value(1.0))
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

/// Independent naive reference for the kept-prefix length: the smallest
/// count whose cumulative raw prior mass reaches `cutoff` of the total
/// (uniform mass per action when the total is not positive), clamped to
/// at least `floor` and at most the list length.
fn reference_kept_count(values: &[f64], cutoff: f64, floor: usize) -> usize {
    let len = values.len();
    if len == 0 {
        return 0;
    }
    let total: f64 = values.iter().sum();
    let reached = |count: usize| {
        if total > 0.0 {
            values[..count].iter().sum::<f64>() >= cutoff * total
        } else {
            count as f64 >= cutoff * len as f64
        }
    };
    let mass_kept = (1..=len).find(|&count| reached(count)).unwrap_or(len);
    mass_kept.max(floor.min(len))
}

// ---------------------------------------------------------------------------
// Truncation laws.
// ---------------------------------------------------------------------------

#[hegel::test(test_cases = 256)]
fn mass_truncation_matches_a_naive_reference(tc: TestCase) {
    let n: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let mask = draw_mask(&tc, n);
    let all_zero: bool = tc.draw(gs::booleans());
    let priors = if all_zero {
        vec![0.0; n]
    } else {
        draw_priors(&tc, n)
    };
    let cap: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let cutoff: f64 = tc.draw(gs::floats::<f64>().min_value(0.01).max_value(1.0));
    let floor: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));

    let full = legal_from_priors(mask, &priors, cap);
    let mut restricted = full.clone();
    truncate_to_prior_mass(&mut restricted, &priors, cutoff, floor);

    let values: Vec<f64> = full.iter().map(|&action| priors[action]).collect();
    let expected = reference_kept_count(&values, cutoff, floor);
    assert_eq!(restricted.len(), expected, "kept count");
    assert_eq!(
        restricted[..],
        full[..expected],
        "the kept actions are the highest-prior prefix"
    );
    assert!(restricted.len() >= floor.min(full.len()), "floor clamp");
    assert!(restricted.len() <= full.len(), "cap clamp");
}

/// With strictly positive priors the cumulative mass over the whole list
/// reaches the total exactly (identical summation order), so a cutoff of
/// 1.0 must keep every action — the anchor behind the end-to-end
/// equivalence law below.
#[hegel::test(test_cases = 256)]
fn full_cutoff_keeps_every_positive_prior_action(tc: TestCase) {
    let n: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let mask = draw_mask(&tc, n);
    let priors = draw_positive_priors(&tc, n);
    let cap: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let floor: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));

    let full = legal_from_priors(mask, &priors, cap);
    let mut restricted = full.clone();
    truncate_to_prior_mass(&mut restricted, &priors, 1.0, floor);
    assert_eq!(restricted, full);
}

/// A single action holding all the prior mass satisfies any cutoff by
/// itself, so the floor alone decides the kept count.
#[hegel::test(test_cases = 256)]
fn a_dominant_prior_prunes_to_the_floor(tc: TestCase) {
    let n: usize = tc.draw(gs::integers::<usize>().min_value(2).max_value(8));
    let hot: usize = tc.draw(gs::integers::<usize>().min_value(0).max_value(n - 1));
    let mut priors = vec![0.0; n];
    priors[hot] = 1.0;
    let floor: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(n));
    let cutoff: f64 = tc.draw(gs::floats::<f64>().min_value(0.01).max_value(1.0));

    let full = legal_from_priors((1 << n) - 1, &priors, n);
    let mut restricted = full.clone();
    truncate_to_prior_mass(&mut restricted, &priors, cutoff, floor);

    assert_eq!(restricted.len(), floor);
    assert_eq!(restricted[0], hot, "the dominant action survives first");
}

/// Exact anchors for the uniform fallback: all-zero priors weight every
/// action equally, so the cutoff keeps a proportional share.
#[test]
fn uniform_fallback_keeps_the_proportional_share() {
    let priors = vec![0.0; 4];
    let full = legal_from_priors(0b1111, &priors, 4);
    assert_eq!(full, vec![0, 1, 2, 3]);

    let mut half = full.clone();
    truncate_to_prior_mass(&mut half, &priors, 0.5, 1);
    assert_eq!(half, vec![0, 1]);

    let mut just_over_half = full.clone();
    truncate_to_prior_mass(&mut just_over_half, &priors, 0.51, 1);
    assert_eq!(just_over_half, vec![0, 1, 2]);

    let mut all = full.clone();
    truncate_to_prior_mass(&mut all, &priors, 1.0, 1);
    assert_eq!(all, vec![0, 1, 2, 3]);
}

/// Cutoff 1.0 with a zero-prior legal action: the zero contributes no
/// mass, so it is dropped — the one documented behavioral difference
/// from disabling the cutoff entirely.
#[test]
fn full_cutoff_drops_zero_prior_actions() {
    let priors = vec![0.5, 0.0, 0.5];
    let full = legal_from_priors(0b111, &priors, 3);
    assert_eq!(full, vec![0, 2, 1]);

    let mut restricted = full.clone();
    truncate_to_prior_mass(&mut restricted, &priors, 1.0, 1);
    assert_eq!(restricted, vec![0, 2]);
}

// ---------------------------------------------------------------------------
// End-to-end laws.
// ---------------------------------------------------------------------------

/// Metamorphic end-to-end law: with strictly positive priors, a cutoff
/// of 1.0 and a floor of 1 keep every legal list unchanged, so the whole
/// search — RNG draws included — must be bitwise identical to pruning
/// disabled.
#[hegel::test(test_cases = 64)]
fn full_cutoff_searches_match_disabled_pruning_bitwise(tc: TestCase) {
    let n: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
    let cells = draw_matrix(&tc, n);
    let player_priors = draw_positive_priors(&tc, n);
    let enemy_priors = draw_positive_priors(&tc, n);
    let value: f64 = tc.draw(gs::floats::<f64>().min_value(-1.0).max_value(1.0));
    let seed: u64 = tc.draw(gs::integers::<u64>());
    let base = JointSearchConfig {
        expansion_budget: tc.draw(gs::integers::<u32>().min_value(1).max_value(16)),
        max_depth: tc.draw(gs::integers::<u32>().min_value(1).max_value(2)),
        regret_iterations: 32,
        ..JointSearchConfig::default()
    };
    let pruned_config = JointSearchConfig {
        prior_mass_cutoff: Some(1.0),
        minimum_actions_per_side: 1,
        ..base.clone()
    };
    let options = SearchOptions {
        sample_actions: tc.draw(gs::booleans()),
        router_score: 1.0,
    };

    let run = |config: JointSearchConfig| {
        let mut provider = MatrixProvider::new(n, cells.clone());
        let root = provider.root();
        let mut evaluator = FixedPriorEvaluator {
            player_priors: player_priors.clone(),
            enemy_priors: enemy_priors.clone(),
            value,
        };
        let mut search = SimultaneousTreeSearch::new(config, seed);
        search.search_with_tree(&mut provider, &mut evaluator, root, options)
    };

    let (off, _) = run(base);
    let (on, _) = run(pruned_config);
    assert_eq!(off, on);
}

/// Structural oracle: random pruned searches uphold every tree
/// invariant, including the pruned legal-list reconstruction. Priors may
/// contain zeros — zero-prior legal actions are dropped by any cutoff,
/// which is exactly the edge worth sweeping.
#[hegel::test]
fn pruned_searches_uphold_every_tree_invariant(tc: TestCase) {
    let staged: bool = tc.draw(gs::booleans());
    let n: usize = if staged {
        2
    } else {
        tc.draw(gs::integers::<usize>().min_value(1).max_value(4))
    };
    let max_actions_per_side: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(13));
    let config = JointSearchConfig {
        prior_mass_cutoff: Some(tc.draw(gs::floats::<f64>().min_value(0.05).max_value(1.0))),
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
    assert_joint_tree_invariants(&tree, &result, &config, "pruned scenario");
}

/// Exact anchor: a 90% cutoff over priors [0.60, 0.35, 0.05] keeps the
/// two-action prefix on both sides, so a root-only search installs a
/// 2x2 grid (4 transitions) instead of 3x3 (9) — the solve-cost lever
/// the cutoff exists for.
#[test]
fn dominated_actions_are_pruned_from_the_root_grid() {
    let priors = vec![0.60, 0.35, 0.05];
    let run = |cutoff: Option<f64>| {
        let config = JointSearchConfig {
            prior_mass_cutoff: cutoff,
            minimum_actions_per_side: 1,
            expansion_budget: 1,
            ..JointSearchConfig::default()
        };
        let mut provider = MatrixProvider::new(3, vec![0.0; 9]);
        let root = provider.root();
        let mut evaluator = FixedPriorEvaluator {
            player_priors: priors.clone(),
            enemy_priors: priors.clone(),
            value: 0.0,
        };
        let mut search = SimultaneousTreeSearch::new(config.clone(), 7);
        let (result, tree) = search.search_with_tree(
            &mut provider,
            &mut evaluator,
            root,
            SearchOptions::default(),
        );
        assert_joint_tree_invariants(&tree, &result, &config, "root pruning anchor");
        (result, tree)
    };

    let (pruned, tree) = run(Some(0.90));
    let root = tree.node(0);
    assert_eq!(root.player_legal, vec![0, 1]);
    assert_eq!(root.enemy_legal, vec![0, 1]);
    assert_eq!(pruned.transitions, 4);
    assert_eq!(pruned.player_policy[2], 0.0);
    assert_eq!(pruned.enemy_policy[2], 0.0);

    let (unpruned, _) = run(None);
    assert_eq!(unpruned.transitions, 9);
}
