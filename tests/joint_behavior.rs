//! End-to-end behavior of the root-only joint search on toy matrix games.
//!
//! Every expected number is derived by hand from the search's equations:
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
    assert_joint_tree_invariants, DivergeAfter, FixedPriorEvaluator, MatrixProvider,
    RecordingProvider, SeedSensitiveProvider, ToySnapshot, TwoStage, UniformEvaluator,
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
    assert_eq!(result.solver, SolverTag::RmPlus);
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
    // Zero learned simulations leave the root matrix untouched, so the
    // final re-solve is skipped and the initial equilibrium stands.
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

/// The opt-in tolerance stops the cold equilibrium at the first
/// interval-aligned checkpoint whose time-average already meets it.
/// Matching pennies from uniform priors sits on the exact fixpoint from
/// iteration one, so the stop fires at the very first check (iteration
/// 64 of the configured 2048) and every output keeps its exact
/// fixpoint value.
#[test]
fn equilibrium_tolerance_stops_at_the_first_checkpoint() {
    let mut provider = MatrixProvider::new(2, vec![1.0, -1.0, -1.0, 1.0]);
    let mut evaluator = UniformEvaluator {
        action_count: 2,
        value: 0.0,
    };
    let config = JointSearchConfig {
        equilibrium_tolerance: Some(0.5),
        ..root_only_config(4)
    };
    let mut search = SimultaneousTreeSearch::new(config.clone(), 7);
    let root = provider.root();
    let (result, tree) = search.search_with_tree(
        &mut provider,
        &mut evaluator,
        root,
        SearchOptions::default(),
    );

    assert_eq!(result.player_policy, vec![0.5, 0.5]);
    assert_eq!(result.enemy_policy, vec![0.5, 0.5]);
    assert_eq!(result.root_value, 0.0);
    assert_eq!(result.exploitability, Some(0.0));
    assert_eq!(result.failure, None);
    let root_diagnostics = result.diagnostics.root.as_ref().expect("root diagnostics");
    assert_eq!(root_diagnostics.equilibrium_iterations, 64);
    assert_joint_tree_invariants(&tree, &result, &config, "tolerance stop");
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
    assert_eq!(result.solver, SolverTag::DivergenceFallback);
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

#[test]
fn deep_search_corrects_a_pessimistic_leaf_estimate() {
    // Root action 0 leads to a stage worth a flat 0.3, action 1 bails at
    // -1. The evaluator insists every live state is worth -0.9, so the
    // root-only equilibrium undervalues action 0; descending into the
    // stage child replaces that estimate with backed-up 0.3 payoffs.
    let mut provider = TwoStage {
        stage_matrix: vec![0.3; 4],
        stage_potential: 0.0,
        bail_value: Some(-1.0),
    };
    let mut evaluator = UniformEvaluator {
        action_count: 2,
        value: -0.9,
    };
    let config = JointSearchConfig {
        expansion_budget: 24,
        minimum_expansion_budget: 24,
        max_depth: 2,
        ..JointSearchConfig::default()
    };
    let mut search = SimultaneousTreeSearch::new(config.clone(), 3);
    let options = SearchOptions {
        sample_actions: false,
        ..SearchOptions::default()
    };
    let (result, tree) =
        search.search_with_tree(&mut provider, &mut evaluator, TwoStage::root(), options);

    assert_eq!(result.failure, None);
    assert_eq!(result.transitions, 24);
    assert_eq!(result.player_action, Some(0));
    assert!(
        result.player_policy[0] > 0.8,
        "player policy: {:?}",
        result.player_policy
    );
    assert!(
        result.root_value > -0.5,
        "root value: {}",
        result.root_value
    );
    assert_eq!(result.diagnostics.deep_search_needed, Some(true));
    assert!(result.diagnostics.tree_simulations > 0);
    assert_eq!(result.diagnostics.tree_max_depth, 1);
    assert_eq!(result.diagnostics.tree_nodes, 2);
    // Every root pair reaches the same stage snapshot: one shared child.
    assert_eq!(tree.nodes.len(), 2);
    assert_eq!(tree.nodes[1].snapshot.id, 1);
    assert!(tree.nodes[1].expanded);
    assert_joint_tree_invariants(&tree, &result, &config, "pessimistic-leaf");
}

#[test]
fn budget_starved_descent_falls_back_to_the_leaf_estimate() {
    // Budget 5 leaves exactly one transition after the 4-transition root
    // install. That simulation resamples its pair (evidence 1 gives
    // resample probability 1), creates the stage child, and must then
    // refuse the child's 4-transition expansion: the child node exists
    // but stays unexpanded, and the learned value is the shaped leaf.
    let mut provider = TwoStage {
        stage_matrix: vec![0.3; 4],
        stage_potential: 0.0,
        bail_value: None,
    };
    let mut evaluator = UniformEvaluator {
        action_count: 2,
        value: -0.9,
    };
    let config = JointSearchConfig {
        expansion_budget: 5,
        minimum_expansion_budget: 5,
        max_depth: 2,
        ..JointSearchConfig::default()
    };
    let mut search = SimultaneousTreeSearch::new(config.clone(), 17);
    let (result, tree) = search.search_with_tree(
        &mut provider,
        &mut evaluator,
        TwoStage::root(),
        SearchOptions::default(),
    );

    assert_eq!(result.transitions, 5);
    assert_eq!(result.diagnostics.tree_simulations, 1);
    assert_eq!(result.diagnostics.tree_nodes, 1);
    assert_eq!(result.diagnostics.tree_max_depth, 0);
    assert_eq!(tree.nodes.len(), 2);
    assert!(!tree.nodes[1].expanded);
    assert_eq!(tree.nodes[1].visits, 0);
    assert_eq!(tree.nodes[1].solve_count, 0);
    // The install solve plus exactly one learned-simulation solve.
    assert_eq!(tree.root().solve_count, 32);
    assert_eq!(tree.root().visits, 1);
    assert_joint_tree_invariants(&tree, &result, &config, "budget-starved");
}

#[test]
fn potential_shapes_live_leaves_but_never_terminal_returns() {
    // Live stage successors carry potential 0.35, so their installed
    // value is clamp(0.9 + 0.35) = 1.0. The bail row ends immediately at
    // 1.7, recorded raw: terminal returns are never shaped or clamped.
    let mut provider = TwoStage {
        stage_matrix: vec![0.0; 4],
        stage_potential: 0.35,
        bail_value: Some(1.7),
    };
    let mut evaluator = UniformEvaluator {
        action_count: 2,
        value: 0.9,
    };
    let config = JointSearchConfig {
        expansion_budget: 4,
        minimum_expansion_budget: 1,
        ..JointSearchConfig::default()
    };
    let mut search = SimultaneousTreeSearch::new(config.clone(), 29);
    let options = SearchOptions {
        sample_actions: false,
        ..SearchOptions::default()
    };
    let (result, tree) =
        search.search_with_tree(&mut provider, &mut evaluator, TwoStage::root(), options);

    assert_eq!(result.transitions, 4);
    assert_eq!(result.payoff_matrix, Some(vec![1.0, 1.0, 1.7, 1.7]));
    assert_eq!(result.payoff_spread, Some(1.7 - 1.0));
    assert_eq!(result.player_action, Some(1));
    assert!(result.root_value > 1.5, "root value: {}", result.root_value);
    let root = tree.root();
    assert_eq!(root.outcomes_at(0, 0)[0].tactical_delta, 0.35);
    assert_eq!(root.outcomes_at(0, 0)[0].leaf_value, 0.9);
    assert_eq!(root.outcomes_at(1, 0)[0].tactical_delta, 0.0);
    assert_eq!(root.outcomes_at(1, 0)[0].leaf_value, 1.7);
    assert_joint_tree_invariants(&tree, &result, &config, "potential-shaping");
}

#[test]
fn chance_resampling_accumulates_evidence_on_seen_pairs() {
    // Every joint pair is terminal with a seed-parity payoff. After the
    // 4-transition install, the descent loop must spend the remaining 8
    // transitions re-drawing outcomes for already-seen pairs, stacking
    // multiple outcomes into the same cells.
    let mut provider = SeedSensitiveProvider;
    let mut evaluator = UniformEvaluator {
        action_count: 2,
        value: 0.0,
    };
    let config = JointSearchConfig {
        expansion_budget: 12,
        minimum_expansion_budget: 12,
        ..JointSearchConfig::default()
    };
    let mut search = SimultaneousTreeSearch::new(config.clone(), 41);
    let (result, tree) = search.search_with_tree(
        &mut provider,
        &mut evaluator,
        SeedSensitiveProvider::root(),
        SearchOptions::default(),
    );

    assert_eq!(result.transitions, 12);
    // Here a simulation learns exactly when it draws a fresh outcome, so
    // the learned count is the 8 post-install transitions.
    assert_eq!(result.diagnostics.tree_simulations, 8);
    assert_eq!(result.diagnostics.tree_max_depth, 0);
    let root = tree.root();
    assert!(root.visits >= 8);
    let fullest_cell = (0..2)
        .flat_map(|player| (0..2).map(move |enemy| root.outcomes_at(player, enemy).len()))
        .max()
        .expect("the root grid is non-empty");
    assert!(fullest_cell >= 2, "12 outcomes over 4 cells must stack");
    assert_joint_tree_invariants(&tree, &result, &config, "chance-resampling");
}

#[test]
fn descent_converges_early_on_a_static_root() {
    // On an all-zero matrix the time-average root policies are exactly
    // uniform after every learned simulation, so the L1 change is 0.0
    // and the stability streak grows by one per learned simulation. The
    // streak reaches the patience of 8 on the 8th learned simulation —
    // each a fresh draw costing one transition — so the loop converges
    // at exactly 4 + 8 = 12 transitions independent of the seed, far
    // short of the budget of 64.
    let mut provider = MatrixProvider::new(2, vec![0.0; 4]);
    let mut evaluator = UniformEvaluator {
        action_count: 2,
        value: 0.0,
    };
    let config = JointSearchConfig {
        expansion_budget: 64,
        minimum_expansion_budget: 8,
        ..JointSearchConfig::default()
    };
    let mut search = SimultaneousTreeSearch::new(config.clone(), 13);
    let root = provider.root();
    let (result, tree) = search.search_with_tree(
        &mut provider,
        &mut evaluator,
        root,
        SearchOptions::default(),
    );

    assert_eq!(result.failure, None);
    assert!(result.diagnostics.tree_converged);
    assert_eq!(result.transitions, 12);
    assert_eq!(result.diagnostics.tree_simulations, 8);
    assert_eq!(result.player_policy, vec![0.5, 0.5]);
    assert_eq!(result.enemy_policy, vec![0.5, 0.5]);
    assert_eq!(result.root_value, 0.0);
    assert_eq!(result.exploitability, Some(0.0));
    assert_eq!(result.diagnostics.deep_search_needed, Some(false));
    // The install solve plus one warm solve per learned simulation.
    assert_eq!(tree.root().solve_count, 144);
    assert_joint_tree_invariants(&tree, &result, &config, "static-convergence");
}

#[test]
fn descent_bails_after_sixty_four_unlearned_simulations() {
    // Once every cell's evidence is deep, the resample probability sits
    // at the 0.1 floor, so roughly nine of ten simulations reuse an
    // existing outcome and learn nothing. Long before the huge budget
    // is spent, 64 consecutive such simulations occur and the loop
    // gives up. Convergence cannot be the exit: the minimum budget
    // equals the never-reached full budget.
    let mut provider = SeedSensitiveProvider;
    let mut evaluator = UniformEvaluator {
        action_count: 2,
        value: 0.0,
    };
    let config = JointSearchConfig {
        expansion_budget: 4000,
        minimum_expansion_budget: 4000,
        ..JointSearchConfig::default()
    };
    let mut search = SimultaneousTreeSearch::new(config.clone(), 21);
    let (result, tree) = search.search_with_tree(
        &mut provider,
        &mut evaluator,
        SeedSensitiveProvider::root(),
        SearchOptions::default(),
    );

    assert_eq!(result.failure, None);
    assert!(!result.diagnostics.tree_converged);
    assert!(
        result.transitions < 4000,
        "the bail must fire before the budget: {}",
        result.transitions
    );
    assert!(result.transitions > 4, "some learning must happen first");
    // Simulations learn exactly when they draw fresh outcomes (cost 1),
    // so the learned count is the post-install transition count.
    assert_eq!(result.diagnostics.tree_simulations, result.transitions - 4);
    assert_joint_tree_invariants(&tree, &result, &config, "unlearned-bail");
}

#[test]
fn mid_descent_divergence_discards_the_inflight_cost() {
    // The provider survives the 4-step root install plus one descent
    // step. The first simulation always resamples (evidence 1), spends
    // that fifth success on a live stage outcome, and then tries to
    // expand the newly created child — whose first step diverges. The
    // failing simulation's in-flight cost is discarded: transitions
    // reports the install only, while the outcome pushed by the
    // successful fifth step stays counted in chance_outcomes.
    let mut provider = DivergeAfter::new(
        TwoStage {
            stage_matrix: vec![1.0, -1.0, -1.0, 1.0],
            stage_potential: 0.0,
            bail_value: None,
        },
        5,
    );
    let mut evaluator = UniformEvaluator {
        action_count: 2,
        value: 0.25,
    };
    let config = JointSearchConfig {
        expansion_budget: 24,
        minimum_expansion_budget: 24,
        max_depth: 2,
        ..JointSearchConfig::default()
    };
    let mut search = SimultaneousTreeSearch::new(config, 1);
    let (result, tree) = search.search_with_tree(
        &mut provider,
        &mut evaluator,
        TwoStage::root(),
        SearchOptions::default(),
    );

    assert_eq!(result.solver, SolverTag::DivergenceFallback);
    assert!(result.failure.is_some());
    assert_eq!(result.player_policy, vec![0.0, 0.0]);
    assert_eq!(result.enemy_policy, vec![0.0, 0.0]);
    assert_eq!(result.player_action, None);
    assert_eq!(result.enemy_action, None);
    assert_eq!(result.root_value, 0.25);
    assert_eq!(result.transitions, 4);
    assert_eq!(result.exploitability, None);
    assert_eq!(result.payoff_matrix, None);

    let diagnostics = &result.diagnostics;
    assert_eq!(diagnostics.tree_simulations, 0);
    assert_eq!(diagnostics.chance_outcomes, 5);
    assert_eq!(diagnostics.tree_nodes, 1);
    assert_eq!(diagnostics.root, None);
    // The root keeps its installed learning; the stage child was created
    // by the diverging simulation but never expanded.
    assert_eq!(tree.nodes.len(), 2);
    assert!(tree.root().expanded);
    assert!(!tree.nodes[1].expanded);
}

#[test]
fn adaptive_routes_shallow_when_depth_is_capped() {
    let mut provider = MatrixProvider::new(2, vec![1.0, -1.0, -1.0, 1.0]);
    let mut evaluator = UniformEvaluator {
        action_count: 2,
        value: 0.0,
    };
    let config = JointSearchConfig {
        adaptive_search: true,
        expansion_budget: 8,
        minimum_expansion_budget: 1,
        ..JointSearchConfig::default()
    };
    let mut search = SimultaneousTreeSearch::new(config.clone(), 3);
    let root = provider.root();
    let (result, tree) = search.search_with_tree(
        &mut provider,
        &mut evaluator,
        root,
        SearchOptions::default(),
    );

    // max_depth 1 wins before every other predicate, including the high
    // default router score.
    assert_eq!(
        result.diagnostics.adaptive_reason,
        AdaptiveReason::ConfiguredRootOnly
    );
    assert!(!result.diagnostics.adaptive_deep_selected);
    assert_eq!(result.diagnostics.adaptive_router_score, 1.0);
    assert_eq!(result.transitions, 4);
    assert_eq!(result.diagnostics.tree_simulations, 0);
    assert_eq!(result.diagnostics.deep_search_needed, None);
    assert_eq!(result.diagnostics.deep_policy_change, 0.0);
    assert!(!result.diagnostics.deep_action_changed);
    assert_eq!(result.player_policy, vec![0.5, 0.5]);
    assert_eq!(result.enemy_policy, vec![0.5, 0.5]);
    assert_eq!(result.exploitability, Some(0.0));
    assert_eq!(tree.root().solve_count, 16);
    assert_joint_tree_invariants(&tree, &result, &config, "adaptive-root-only");
}

#[test]
fn adaptive_routes_deep_on_a_high_router_score() {
    let mut provider = TwoStage {
        stage_matrix: vec![1.0, -1.0, -1.0, 1.0],
        stage_potential: 0.0,
        bail_value: None,
    };
    let mut evaluator = UniformEvaluator {
        action_count: 2,
        value: 0.0,
    };
    let config = JointSearchConfig {
        adaptive_search: true,
        expansion_budget: 16,
        minimum_expansion_budget: 16,
        max_depth: 2,
        ..JointSearchConfig::default()
    };
    let mut search = SimultaneousTreeSearch::new(config.clone(), 5);
    let (result, tree) = search.search_with_tree(
        &mut provider,
        &mut evaluator,
        TwoStage::root(),
        SearchOptions::default(),
    );

    // The default router score of 1.0 clears the 0.55 threshold.
    assert_eq!(
        result.diagnostics.adaptive_reason,
        AdaptiveReason::LearnedRouter
    );
    assert!(result.diagnostics.adaptive_deep_selected);
    assert_eq!(result.diagnostics.adaptive_router_score, 1.0);
    assert_eq!(result.transitions, 16);
    assert!(result.diagnostics.tree_simulations > 0);
    assert!(result.diagnostics.deep_search_needed.is_some());
    assert_joint_tree_invariants(&tree, &result, &config, "adaptive-router-deep");
}

#[test]
fn adaptive_routes_deep_on_online_exploitability() {
    // Warm-solving matching pennies from lopsided priors leaves the
    // 16th RM+ iterate far from equilibrium, so the online
    // exploitability clears the 0.08 threshold and forces deep search
    // even with a cold router.
    let mut provider = MatrixProvider::new(2, vec![1.0, -1.0, -1.0, 1.0]);
    let mut evaluator = FixedPriorEvaluator {
        player_priors: vec![0.9, 0.1],
        enemy_priors: vec![0.9, 0.1],
        value: 0.0,
    };
    let config = JointSearchConfig {
        adaptive_search: true,
        expansion_budget: 8,
        minimum_expansion_budget: 8,
        max_depth: 2,
        ..JointSearchConfig::default()
    };
    let mut search = SimultaneousTreeSearch::new(config.clone(), 7);
    let root = provider.root();
    let options = SearchOptions {
        router_score: 0.0,
        ..SearchOptions::default()
    };
    let (result, tree) = search.search_with_tree(&mut provider, &mut evaluator, root, options);

    assert_eq!(
        result.diagnostics.adaptive_reason,
        AdaptiveReason::RootOnlineExploitability
    );
    assert!(result.diagnostics.adaptive_deep_selected);
    assert_eq!(result.diagnostics.adaptive_router_score, 0.0);
    let root_diagnostics = result.diagnostics.root.as_ref().expect("root diagnostics");
    assert!(
        root_diagnostics.online_exploitability >= 0.08,
        "online exploitability: {}",
        root_diagnostics.online_exploitability
    );
    assert_eq!(result.transitions, 8);
    assert_joint_tree_invariants(&tree, &result, &config, "adaptive-exploitability");
}

#[test]
fn adaptive_routes_deep_on_payoff_uncertainty() {
    // Uniform-prior matching pennies warm-solves to exactly the uniform
    // fixpoint, so the online exploitability of 0.0 skips the previous
    // predicate. The payoff spread of 2 and the two-sided policy entropy
    // of 2·ln 2 then trip the uncertainty predicate.
    let mut provider = MatrixProvider::new(2, vec![1.0, -1.0, -1.0, 1.0]);
    let mut evaluator = UniformEvaluator {
        action_count: 2,
        value: 0.0,
    };
    let config = JointSearchConfig {
        adaptive_search: true,
        expansion_budget: 8,
        minimum_expansion_budget: 8,
        max_depth: 2,
        ..JointSearchConfig::default()
    };
    let mut search = SimultaneousTreeSearch::new(config.clone(), 9);
    let root = provider.root();
    let options = SearchOptions {
        router_score: 0.0,
        ..SearchOptions::default()
    };
    let (result, tree) = search.search_with_tree(&mut provider, &mut evaluator, root, options);

    assert_eq!(
        result.diagnostics.adaptive_reason,
        AdaptiveReason::RootPayoffUncertainty
    );
    assert!(result.diagnostics.adaptive_deep_selected);
    let root_diagnostics = result.diagnostics.root.as_ref().expect("root diagnostics");
    assert_eq!(root_diagnostics.online_exploitability, 0.0);
    assert_eq!(result.transitions, 8);
    assert_joint_tree_invariants(&tree, &result, &config, "adaptive-uncertainty");
}

#[test]
fn adaptive_forces_a_calibration_deep_sample() {
    // A constant matrix defeats every informative predicate; a forced
    // calibration fraction of 1.0 turns the budget coin into a
    // certainty, since the coin is always below 1.
    let mut provider = MatrixProvider::new(2, vec![0.0; 4]);
    let mut evaluator = UniformEvaluator {
        action_count: 2,
        value: 0.0,
    };
    let config = JointSearchConfig {
        adaptive_search: true,
        adaptive_force_deep_fraction: 1.0,
        expansion_budget: 8,
        minimum_expansion_budget: 8,
        max_depth: 2,
        ..JointSearchConfig::default()
    };
    let mut search = SimultaneousTreeSearch::new(config.clone(), 11);
    let root = provider.root();
    let options = SearchOptions {
        router_score: 0.0,
        ..SearchOptions::default()
    };
    let (result, tree) = search.search_with_tree(&mut provider, &mut evaluator, root, options);

    assert_eq!(
        result.diagnostics.adaptive_reason,
        AdaptiveReason::ForcedCalibrationSample
    );
    assert!(result.diagnostics.adaptive_deep_selected);
    assert_eq!(result.transitions, 8);
    assert_joint_tree_invariants(&tree, &result, &config, "adaptive-forced");
}

#[test]
fn adaptive_stays_shallow_on_a_stable_root() {
    // The same constant matrix with the forced fraction at 0.0: every
    // deep predicate falls through and the root stays shallow, keeping
    // the initial equilibrium as the result untouched.
    let mut provider = MatrixProvider::new(2, vec![0.0; 4]);
    let mut evaluator = UniformEvaluator {
        action_count: 2,
        value: 0.0,
    };
    let config = JointSearchConfig {
        adaptive_search: true,
        adaptive_force_deep_fraction: 0.0,
        expansion_budget: 8,
        minimum_expansion_budget: 1,
        max_depth: 2,
        ..JointSearchConfig::default()
    };
    let mut search = SimultaneousTreeSearch::new(config.clone(), 11);
    let root = provider.root();
    let options = SearchOptions {
        router_score: 0.0,
        ..SearchOptions::default()
    };
    let (result, tree) = search.search_with_tree(&mut provider, &mut evaluator, root, options);

    assert_eq!(
        result.diagnostics.adaptive_reason,
        AdaptiveReason::RouterStableRoot
    );
    assert!(!result.diagnostics.adaptive_deep_selected);
    assert_eq!(result.transitions, 4);
    assert_eq!(result.diagnostics.tree_simulations, 0);
    assert_eq!(result.diagnostics.deep_search_needed, None);
    assert_eq!(result.player_policy, vec![0.5, 0.5]);
    assert_eq!(result.enemy_policy, vec![0.5, 0.5]);
    assert_eq!(tree.root().solve_count, 16);
    assert_joint_tree_invariants(&tree, &result, &config, "adaptive-stable");
}

#[test]
fn same_seed_deep_runs_are_identical() {
    let run_once = || {
        let mut provider = TwoStage {
            stage_matrix: vec![1.0, -1.0, -1.0, 1.0],
            stage_potential: 0.15,
            bail_value: Some(-1.0),
        };
        let mut evaluator = FixedPriorEvaluator {
            player_priors: vec![0.6, 0.4],
            enemy_priors: vec![0.45, 0.55],
            value: -0.2,
        };
        let config = JointSearchConfig {
            expansion_budget: 24,
            minimum_expansion_budget: 24,
            max_depth: 3,
            ..JointSearchConfig::default()
        };
        let mut search = SimultaneousTreeSearch::new(config, 31);
        search.search(
            &mut provider,
            &mut evaluator,
            TwoStage::root(),
            SearchOptions::default(),
        )
    };
    let first = run_once();
    assert_eq!(first.failure, None);
    assert!(first.diagnostics.tree_simulations > 0);
    assert_eq!(run_once(), first);
}
