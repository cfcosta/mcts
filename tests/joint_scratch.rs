//! Scratch-reuse laws: threading caller-held buffers through the solver
//! never changes an output bit.
//!
//! [`solve_node`] is defined as [`solve_node_with_scratch`] over a fresh
//! scratch, so every existing solver characterization already pins the
//! shared body. The laws here pin the only genuinely new behavior: a
//! scratch left dirty by earlier solves — of any shape, any flags, any
//! iteration counts — is indistinguishable from a fresh one on every
//! solver-written field, and the `_into` twins of the policy helpers
//! write into dirty buffers exactly what their allocating originals
//! return.

use std::collections::HashMap;

use hegel::{generators as gs, TestCase};
use mcts_rs::joint::{
    average_policy, average_policy_into, mixed_policy, mixed_policy_into, solve_node,
    solve_node_with_scratch, Outcome, SolveScratch, TreeNode,
};

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// A bare node over `action_count` actions, shaped exactly as `make_node`
/// builds one: empty policies until the first solve, zeroed solver state,
/// empty outcome cells.
fn blank_node(
    action_count: usize,
    player_legal: Vec<usize>,
    enemy_legal: Vec<usize>,
    player_priors: Vec<f64>,
    enemy_priors: Vec<f64>,
) -> TreeNode<()> {
    let cells = action_count * action_count;
    let outcomes: Vec<Vec<Outcome<()>>> = (0..cells).map(|_| Vec::new()).collect();
    TreeNode {
        snapshot: (),
        player_priors,
        enemy_priors,
        leaf_value: 0.0,
        player_legal,
        enemy_legal,
        payoff: vec![0.0; cells],
        counts: vec![0; cells],
        outcomes,
        children: HashMap::new(),
        player_policy: Vec::new(),
        enemy_policy: Vec::new(),
        root_value: 0.0,
        exploitability: 0.0,
        online_exploitability: 0.0,
        expanded: false,
        visits: 0,
        solve_count: 0,
        player_strategy_sum: vec![0.0; action_count],
        enemy_strategy_sum: vec![0.0; action_count],
        player_regrets: vec![0.0; action_count],
        enemy_regrets: vec![0.0; action_count],
    }
}

/// Draws a non-empty legal subset of `0..action_count`.
fn draw_legal(tc: &TestCase, action_count: usize) -> Vec<usize> {
    let mut legal: Vec<usize> = (0..action_count)
        .filter(|_| tc.draw(gs::booleans()))
        .collect();
    if legal.is_empty() {
        legal.push(
            tc.draw(
                gs::integers::<usize>()
                    .min_value(0)
                    .max_value(action_count - 1),
            ),
        );
    }
    legal
}

/// Priors for one side; a mass-free side exercises the uniform fallback
/// of the prior normalization inside the solve.
fn draw_priors(tc: &TestCase, action_count: usize) -> Vec<f64> {
    if tc.draw(gs::booleans()) {
        return vec![0.0; action_count];
    }
    tc.draw(
        gs::vecs(gs::floats::<f64>().min_value(0.05).max_value(1.0))
            .min_size(action_count)
            .max_size(action_count),
    )
}

fn draw_index(tc: &TestCase, len: usize) -> usize {
    tc.draw(gs::integers::<usize>().min_value(0).max_value(len - 1))
}

/// Every field a node solve writes, compared bit for bit.
fn assert_solver_state_identical(plain: &TreeNode<()>, scratched: &TreeNode<()>, context: &str) {
    assert_eq!(
        bits(&plain.player_policy),
        bits(&scratched.player_policy),
        "player policy ({context})"
    );
    assert_eq!(
        bits(&plain.enemy_policy),
        bits(&scratched.enemy_policy),
        "enemy policy ({context})"
    );
    assert_eq!(
        bits(&plain.player_regrets),
        bits(&scratched.player_regrets),
        "player regrets ({context})"
    );
    assert_eq!(
        bits(&plain.enemy_regrets),
        bits(&scratched.enemy_regrets),
        "enemy regrets ({context})"
    );
    assert_eq!(
        bits(&plain.player_strategy_sum),
        bits(&scratched.player_strategy_sum),
        "player strategy sum ({context})"
    );
    assert_eq!(
        bits(&plain.enemy_strategy_sum),
        bits(&scratched.enemy_strategy_sum),
        "enemy strategy sum ({context})"
    );
    assert_eq!(
        plain.root_value.to_bits(),
        scratched.root_value.to_bits(),
        "root value ({context})"
    );
    assert_eq!(
        plain.exploitability.to_bits(),
        scratched.exploitability.to_bits(),
        "exploitability ({context})"
    );
    assert_eq!(
        plain.solve_count, scratched.solve_count,
        "solve count ({context})"
    );
}

/// A dirty scratch is bitwise-equivalent to a fresh one: evolving two
/// identical nodes through the same record/solve history — one via
/// [`solve_node`], one via [`solve_node_with_scratch`] over a scratch
/// shared across every round and both shapes — leaves identical solver
/// state after every solve.
#[hegel::test(test_cases = 40)]
fn dirty_scratch_solves_match_fresh_solves_bitwise(tc: TestCase) {
    let mut scratch = SolveScratch::default();
    for shape in 0..2u32 {
        let action_count: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
        let player_legal = draw_legal(&tc, action_count);
        let enemy_legal = draw_legal(&tc, action_count);
        let player_priors = draw_priors(&tc, action_count);
        let enemy_priors = draw_priors(&tc, action_count);
        let mut plain = blank_node(
            action_count,
            player_legal.clone(),
            enemy_legal.clone(),
            player_priors.clone(),
            enemy_priors.clone(),
        );
        let mut scratched = blank_node(
            action_count,
            player_legal.clone(),
            enemy_legal.clone(),
            player_priors,
            enemy_priors,
        );
        let rounds: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(3));
        for round in 0..rounds {
            let records: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
            for _ in 0..records {
                let player = player_legal[draw_index(&tc, player_legal.len())];
                let enemy = enemy_legal[draw_index(&tc, enemy_legal.len())];
                let value: f64 = tc.draw(gs::floats::<f64>().min_value(-1.0).max_value(1.0));
                plain.record_value(player, enemy, value);
                scratched.record_value(player, enemy, value);
            }
            let iterations: u32 = tc.draw(gs::integers::<u32>().min_value(1).max_value(24));
            let average_policies: bool = tc.draw(gs::booleans());
            let cfr_plus: bool = tc.draw(gs::booleans());
            solve_node(&mut plain, iterations, average_policies, cfr_plus);
            solve_node_with_scratch(
                &mut scratched,
                iterations,
                average_policies,
                cfr_plus,
                &mut scratch,
            );
            assert_solver_state_identical(
                &plain,
                &scratched,
                &format!("shape {shape}, round {round}"),
            );
        }
    }
}

/// The `_into` policy helpers write into a dirty buffer of any prior
/// length exactly what their allocating originals return, on every
/// branch (positive and mass-free priors, weighted and fallback
/// averages).
#[hegel::test(test_cases = 60)]
fn into_helpers_match_their_allocating_twins_on_dirty_buffers(tc: TestCase) {
    let action_count: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let legal = draw_legal(&tc, action_count);
    let priors = draw_priors(&tc, action_count);
    let policy: Vec<f64> = tc.draw(
        gs::vecs(gs::floats::<f64>().min_value(0.0).max_value(1.0))
            .min_size(action_count)
            .max_size(action_count),
    );
    let visits: u32 = tc.draw(gs::integers::<u32>().min_value(0).max_value(500));
    let exploration: f64 = tc.draw(gs::floats::<f64>().min_value(0.0).max_value(0.5));
    let mut buffer: Vec<f64> =
        tc.draw(gs::vecs(gs::floats::<f64>().min_value(-9.0).max_value(9.0)).max_size(12));

    let expected = mixed_policy(&policy, &priors, &legal, visits, exploration);
    mixed_policy_into(&policy, &priors, &legal, visits, exploration, &mut buffer);
    assert_eq!(
        bits(&expected),
        bits(&buffer),
        "mixed_policy_into must match mixed_policy"
    );

    // The buffer now carries the mixed output — genuinely dirty input
    // for the average twin.
    let strategy_sum: Vec<f64> = tc.draw(
        gs::vecs(gs::floats::<f64>().min_value(0.0).max_value(64.0))
            .min_size(action_count)
            .max_size(action_count),
    );
    let total_weight = match draw_index(&tc, 3) {
        0 => 0.0,
        1 => -1.0,
        _ => tc.draw(gs::floats::<f64>().min_value(0.1).max_value(64.0)),
    };
    let expected = average_policy(&strategy_sum, total_weight, &policy);
    average_policy_into(&strategy_sum, total_weight, &policy, &mut buffer);
    assert_eq!(
        bits(&expected),
        bits(&buffer),
        "average_policy_into must match average_policy"
    );
}
