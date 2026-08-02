//! Structural invariants of the joint search across a configuration grid.
//!
//! Every combination runs a full `search_with_tree` and hands the tree,
//! result, and configuration to `assert_joint_tree_invariants`, which
//! re-derives everything checkable from first principles: legal-list
//! construction, solver-state consistency, matrix-cell bookkeeping,
//! counter identities, budget bounds, and the result surface. The grid
//! deliberately crosses budgets that starve, exactly cover, and exceed
//! the root install with depths that disable, allow, and stack descent.

mod support;

use mcts_rs::joint::{JointSearchConfig, SearchOptions, SimultaneousTreeSearch, TransitionProvider};
use support::joint::{
    assert_joint_tree_invariants, FixedPriorEvaluator, SeedSensitiveProvider, ToySnapshot, TwoStage,
};

/// Runs one search and checks every invariant. The asymmetric priors put
/// the enemy legal list in reversed order (action 1 first), so ordering
/// bugs cannot hide behind symmetric fixtures.
fn run_case<P>(
    mut provider: P,
    root: ToySnapshot,
    config: JointSearchConfig,
    seed: u64,
    options: SearchOptions,
    ctx: &str,
) where
    P: TransitionProvider<Snapshot = ToySnapshot>,
{
    let mut evaluator = FixedPriorEvaluator {
        player_priors: vec![0.6, 0.4],
        enemy_priors: vec![0.45, 0.55],
        value: -0.2,
    };
    let mut search = SimultaneousTreeSearch::new(config, seed);
    let (result, tree) = search.search_with_tree(&mut provider, &mut evaluator, root, options);
    assert!(result.failure.is_none(), "{ctx}: unexpected divergence");
    assert_joint_tree_invariants(&tree, &result, search.config(), ctx);
}

#[test]
fn invariants_hold_across_the_configuration_grid() {
    for &expansion_budget in &[1u32, 4, 24] {
        for &max_depth in &[1u32, 2, 3] {
            for &chance_samples_per_joint in &[1u32, 2] {
                for &adaptive_search in &[false, true] {
                    for &sample_actions in &[true, false] {
                        for &seed in &[0u64, 7] {
                            let config = JointSearchConfig {
                                expansion_budget,
                                // Convergence cannot stop the descent
                                // loop before the budget: it requires
                                // the minimum, which equals the budget.
                                minimum_expansion_budget: expansion_budget,
                                max_depth,
                                chance_samples_per_joint,
                                adaptive_search,
                                ..JointSearchConfig::default()
                            };
                            // The default router score of 1.0 routes
                            // adaptive runs deep whenever the depth
                            // allows, so both descent paths appear.
                            let options = SearchOptions {
                                sample_actions,
                                ..SearchOptions::default()
                            };
                            let ctx = format!(
                                "budget {expansion_budget} depth {max_depth} \
                                 samples {chance_samples_per_joint} \
                                 adaptive {adaptive_search} \
                                 sampled {sample_actions} seed {seed}"
                            );
                            run_case(
                                TwoStage {
                                    stage_matrix: vec![1.0, -1.0, -1.0, 1.0],
                                    stage_potential: 0.15,
                                    bail_value: Some(-1.0),
                                },
                                TwoStage::root(),
                                config.clone(),
                                seed,
                                options,
                                &format!("two-stage: {ctx}"),
                            );
                            run_case(
                                SeedSensitiveProvider,
                                SeedSensitiveProvider::root(),
                                config,
                                seed,
                                options,
                                &format!("seed-parity: {ctx}"),
                            );
                        }
                    }
                }
            }
        }
    }
}
