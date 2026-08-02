//! End-to-end behavior of the root-only joint search on toy matrix games.
//!
//! Every expected number is derived by hand from the ported equations:
//! regret matching from uniform priors on matching pennies is an exact
//! fixpoint, and on a dominant-strategy game the pure equilibrium locks
//! in at iteration 2, leaving exactly half an iteration of prior mass in
//! the 2048-iteration time-average. All quantities involved are dyadic
//! rationals, so assertions are exact unless noted.

mod support;

use mcts_rs::joint::{
    AdaptiveReason, JointSearchConfig, RootDiagnostics, SearchOptions, SimultaneousTreeSearch,
    SolverTag,
};
use support::joint::{
    DivergeAfter, FixedPriorEvaluator, MatrixProvider, RecordingProvider, ToySnapshot,
    UniformEvaluator,
};

/// A config whose descent budget the root install always consumes, so the
/// root-only search already exhibits its final semantics.
fn root_only_config(expansion_budget: u32) -> JointSearchConfig {
    JointSearchConfig {
        expansion_budget,
        minimum_expansion_budget: 1,
        ..JointSearchConfig::default()
    }
}

#[test]
fn root_only_search_solves_the_installed_matrix() {
    let mut provider = MatrixProvider::new(2, vec![1.0, -1.0, -1.0, 1.0]);
    let mut evaluator = UniformEvaluator {
        action_count: 2,
        value: 0.0,
    };
    let mut search = SimultaneousTreeSearch::new(root_only_config(4), 7);
    let root = provider.root();
    let (result, tree) = search.search_with_tree(
        &mut provider,
        &mut evaluator,
        root,
        SearchOptions::default(),
    );

    // Matching pennies from uniform priors: regrets never turn positive,
    // so every iterate is the prior and all outputs are exact.
    assert_eq!(result.player_policy, vec![0.5, 0.5]);
    assert_eq!(result.enemy_policy, vec![0.5, 0.5]);
    assert_eq!(result.root_value, 0.0);
    assert_eq!(result.exploitability, Some(0.0));
    assert_eq!(result.transitions, 4);
    assert_eq!(result.solver, SolverTag::RmPlusPooledNodeV3);
    assert_eq!(result.payoff_matrix, Some(vec![1.0, -1.0, -1.0, 1.0]));
    assert_eq!(result.payoff_spread, Some(2.0));
    assert_eq!(result.failure, None);
    assert!(matches!(result.player_action, Some(0 | 1)));
    assert!(matches!(result.enemy_action, Some(0 | 1)));

    let diagnostics = &result.diagnostics;
    assert_eq!(diagnostics.tree_nodes, 1);
    assert_eq!(diagnostics.tree_simulations, 0);
    assert_eq!(diagnostics.tree_max_depth, 0);
    assert_eq!(diagnostics.chance_outcomes, 4);
    assert_eq!(diagnostics.sampled_joint_coverage, 1.0);
    assert!(!diagnostics.tree_converged);
    assert!(diagnostics.adaptive_deep_selected);
    assert_eq!(diagnostics.adaptive_router_score, 1.0);
    assert_eq!(diagnostics.adaptive_reason, AdaptiveReason::Disabled);
    // The final equilibrium re-solves an unchanged matrix, so it equals
    // the initial one bitwise.
    assert_eq!(diagnostics.deep_policy_change, 0.0);
    assert!(!diagnostics.deep_action_changed);
    assert_eq!(diagnostics.deep_search_needed, Some(false));
    assert_eq!(
        diagnostics.root,
        Some(RootDiagnostics {
            joint_actions: 4,
            solves: 16,
            online_exploitability: 0.0,
            final_exploitability: 0.0,
            equilibrium_iterations: 2048,
        })
    );

    let node = tree.root();
    assert!(node.expanded);
    assert_eq!(node.solve_count, 16);
    for player in 0..2 {
        for enemy in 0..2 {
            assert_eq!(node.count_at(player, enemy), 1);
            assert_eq!(node.outcomes_at(player, enemy).len(), 1);
        }
    }
}

#[test]
fn deterministic_actions_argmax_the_dominant_equilibrium() {
    // Payoff [[2, 1], [0, -1]]: player action 0 and enemy action 1 are
    // strictly dominant. RM+ locks the pure equilibrium at iteration 2,
    // so the time-average keeps exactly half an iteration of prior mass:
    // 2047.5/2048 on the dominant action.
    let dominant = 2047.5 / 2048.0;
    let residue = 0.5 / 2048.0;
    let mut provider = MatrixProvider::new(2, vec![2.0, 1.0, 0.0, -1.0]);
    let mut evaluator = UniformEvaluator {
        action_count: 2,
        value: 0.0,
    };
    let mut search = SimultaneousTreeSearch::new(root_only_config(4), 11);
    let root = provider.root();
    let options = SearchOptions {
        sample_actions: false,
        ..SearchOptions::default()
    };
    let result = search.search(&mut provider, &mut evaluator, root, options);

    assert_eq!(result.player_action, Some(0));
    assert_eq!(result.enemy_action, Some(1));
    assert_eq!(result.player_policy, vec![dominant, residue]);
    assert_eq!(result.enemy_policy, vec![residue, dominant]);
    // On the averages: value = p·M·e = 2047.5/2048; the best responses
    // against them leave an exploitability of exactly 3/4096.
    assert_eq!(result.root_value, dominant);
    assert_eq!(result.exploitability, Some(3.0 / 4096.0));
    assert_eq!(result.payoff_spread, Some(3.0));
    assert_eq!(result.payoff_matrix, Some(vec![2.0, 1.0, 0.0, -1.0]));
    assert_eq!(result.diagnostics.deep_search_needed, Some(false));
}

#[test]
fn chance_seeds_are_shared_across_pairs() {
    // Common random numbers: seeds are drawn once per sample index and
    // reused across every joint pair, in pair-major sample-minor order.
    let mut provider = RecordingProvider::new(MatrixProvider::new(2, vec![0.0; 4]));
    let mut evaluator = UniformEvaluator {
        action_count: 2,
        value: 0.0,
    };
    let config = JointSearchConfig {
        chance_samples_per_joint: 2,
        ..root_only_config(8)
    };
    let mut search = SimultaneousTreeSearch::new(config, 5);
    let root = provider.inner.root();
    let result = search.search(
        &mut provider,
        &mut evaluator,
        root,
        SearchOptions::default(),
    );

    assert_eq!(result.transitions, 8);
    assert_eq!(provider.log.len(), 8);
    let seeds = [provider.log[0].3, provider.log[1].3];
    assert_ne!(seeds[0], seeds[1]);
    let expected_pairs = [
        (0, 0),
        (0, 0),
        (0, 1),
        (0, 1),
        (1, 0),
        (1, 0),
        (1, 1),
        (1, 1),
    ];
    for (index, &(parent, player, enemy, seed)) in provider.log.iter().enumerate() {
        assert_eq!(parent, 0);
        assert_eq!((player, enemy), expected_pairs[index]);
        assert_eq!(seed, seeds[index % 2]);
    }
    // Eight samples collapse to four distinct pairs over a four-pair grid.
    assert_eq!(result.diagnostics.chance_outcomes, 8);
    assert_eq!(result.diagnostics.sampled_joint_coverage, 1.0);
}

#[test]
fn provider_divergence_yields_the_fallback_result() {
    let mut provider = DivergeAfter::new(MatrixProvider::new(2, vec![1.0, -1.0, -1.0, 1.0]), 2);
    let mut evaluator = FixedPriorEvaluator {
        player_priors: vec![0.7, 0.3],
        enemy_priors: vec![0.4, 0.6],
        value: 0.25,
    };
    let mut search = SimultaneousTreeSearch::new(root_only_config(4), 3);
    let root = provider.inner.root();
    let (result, tree) = search.search_with_tree(
        &mut provider,
        &mut evaluator,
        root,
        SearchOptions::default(),
    );

    // The fallback ignores the priors: policies are zeroed, only the
    // prior value survives, and the failing step is counted.
    assert_eq!(result.player_policy, vec![0.0, 0.0]);
    assert_eq!(result.enemy_policy, vec![0.0, 0.0]);
    assert_eq!(result.player_action, None);
    assert_eq!(result.enemy_action, None);
    assert_eq!(result.root_value, 0.25);
    assert_eq!(result.transitions, 3);
    assert_eq!(result.solver, SolverTag::DivergenceFallbackV1);
    assert_eq!(result.exploitability, None);
    assert_eq!(result.payoff_spread, None);
    assert_eq!(result.payoff_matrix, None);
    assert!(result.failure.is_some());

    let diagnostics = &result.diagnostics;
    assert_eq!(diagnostics.tree_nodes, 0);
    assert_eq!(diagnostics.chance_outcomes, 0);
    assert_eq!(diagnostics.sampled_joint_coverage, 0.0);
    assert_eq!(diagnostics.adaptive_reason, AdaptiveReason::Disabled);
    assert_eq!(diagnostics.deep_search_needed, None);
    assert_eq!(diagnostics.root, None);

    // The root node exists but the failed install never touched it.
    assert_eq!(tree.nodes.len(), 1);
    assert!(!tree.root().expanded);
    assert_eq!(tree.root().solve_count, 0);
}

#[test]
#[should_panic(expected = "cannot search a terminal state")]
fn searching_a_terminal_root_panics() {
    let mut provider = MatrixProvider::new(2, vec![0.0; 4]);
    let mut evaluator = UniformEvaluator {
        action_count: 2,
        value: 0.0,
    };
    let mut search = SimultaneousTreeSearch::new(root_only_config(4), 1);
    let root = ToySnapshot::terminal(0, 1.0);
    search.search(
        &mut provider,
        &mut evaluator,
        root,
        SearchOptions::default(),
    );
}

#[test]
fn action_cap_restricts_the_grid_to_the_top_priors() {
    let mut provider = MatrixProvider::new(2, vec![0.4, 0.0, 0.0, 0.0]);
    let mut evaluator = FixedPriorEvaluator {
        player_priors: vec![0.9, 0.1],
        enemy_priors: vec![0.9, 0.1],
        value: 0.0,
    };
    let config = JointSearchConfig {
        max_actions_per_side: 1,
        ..root_only_config(1)
    };
    let mut search = SimultaneousTreeSearch::new(config, 9);
    let root = provider.root();
    let result = search.search(
        &mut provider,
        &mut evaluator,
        root,
        SearchOptions::default(),
    );

    // Only the top-prior action survives on each side: a 1x1 grid whose
    // single cell is the whole equilibrium.
    assert_eq!(result.transitions, 1);
    assert_eq!(result.player_policy, vec![1.0, 0.0]);
    assert_eq!(result.enemy_policy, vec![1.0, 0.0]);
    assert_eq!(result.player_action, Some(0));
    assert_eq!(result.enemy_action, Some(0));
    assert_eq!(result.root_value, 0.4);
    assert_eq!(result.exploitability, Some(0.0));
    assert_eq!(result.payoff_spread, Some(0.0));
    assert_eq!(result.payoff_matrix, Some(vec![0.4, 0.0, 0.0, 0.0]));
    assert_eq!(result.diagnostics.root.as_ref().unwrap().joint_actions, 1);
}

#[test]
fn same_seed_runs_are_identical() {
    let run_once = |with_tree: bool| {
        let mut provider = MatrixProvider::new(2, vec![0.3, -0.8, -0.5, 0.9]);
        let mut evaluator = FixedPriorEvaluator {
            player_priors: vec![0.6, 0.4],
            enemy_priors: vec![0.5, 0.5],
            value: 0.1,
        };
        let mut search = SimultaneousTreeSearch::new(root_only_config(4), 42);
        let root = provider.root();
        if with_tree {
            search
                .search_with_tree(
                    &mut provider,
                    &mut evaluator,
                    root,
                    SearchOptions::default(),
                )
                .0
        } else {
            search.search(
                &mut provider,
                &mut evaluator,
                root,
                SearchOptions::default(),
            )
        }
    };
    let first = run_once(false);
    let second = run_once(false);
    assert_eq!(first, second);
    // search() is exactly search_with_tree() with the tree dropped.
    assert_eq!(run_once(true), first);
}

#[test]
#[should_panic(expected = "invalid search config")]
fn constructing_with_an_invalid_config_panics() {
    let config = JointSearchConfig {
        expansion_budget: 0,
        ..JointSearchConfig::default()
    };
    SimultaneousTreeSearch::new(config, 0);
}
