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
fn run_case<P>(mut provider: P, root: ToySnapshot, config: JointSearchConfig, seed: u64, ctx: &str)
where
    P: TransitionProvider<Snapshot = ToySnapshot>,
{
    let mut evaluator = FixedPriorEvaluator {
        player_priors: vec![0.6, 0.4],
        enemy_priors: vec![0.45, 0.55],
        value: -0.2,
    };
    let mut search = SimultaneousTreeSearch::new(config, seed);
    let (result, tree) = search.search_with_tree(
        &mut provider,
        &mut evaluator,
        root,
        SearchOptions::default(),
    );
    assert!(result.failure.is_none(), "{ctx}: unexpected divergence");
    assert_joint_tree_invariants(&tree, &result, search.config(), ctx);
}

#[test]
fn invariants_hold_across_the_configuration_grid() {
    for &expansion_budget in &[1u32, 4, 24] {
        for &max_depth in &[1u32, 2, 3] {
            for &chance_samples_per_joint in &[1u32, 2] {
                for &seed in &[0u64, 7] {
                    let config = JointSearchConfig {
                        expansion_budget,
                        // Convergence cannot stop the descent loop before
                        // the budget, keeping the grid stable as later
                        // milestones add the early-exit machinery.
                        minimum_expansion_budget: expansion_budget,
                        max_depth,
                        chance_samples_per_joint,
                        ..JointSearchConfig::default()
                    };
                    let ctx = format!(
                        "budget {expansion_budget} depth {max_depth} \
                         samples {chance_samples_per_joint} seed {seed}"
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
                        &format!("two-stage: {ctx}"),
                    );
                    run_case(
                        SeedSensitiveProvider,
                        SeedSensitiveProvider::root(),
                        config,
                        seed,
                        &format!("seed-parity: {ctx}"),
                    );
                }
            }
        }
    }
}
