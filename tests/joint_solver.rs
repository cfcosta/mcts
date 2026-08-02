//! Characterization tests for the cold RM+ solver, ported from the Python
//! suite. Exact equalities are asserted only where every intermediate is
//! exactly representable; accumulated quantities use tolerances because
//! this port sums sequentially while NumPy sums pairwise.

use hegel::{generators as gs, TestCase};
use mcts_rs::joint::{
    normalized_prior, solve_zero_sum_regret, solve_zero_sum_regret_with_tolerance,
    EQUILIBRIUM_CHECK_INTERVAL,
};

/// Matching pennies: with uniform priors every RM+ iterate is the exact
/// fixpoint (row values are exactly zero, regrets never grow), so the
/// averaged output is exact.
#[test]
fn matching_pennies_solves_to_the_uniform_fixpoint() {
    let payoff = [1.0, -1.0, -1.0, 1.0];
    let (player, enemy, value, exploitability) = solve_zero_sum_regret(
        &payoff,
        2,
        &[0.5, 0.5],
        &[0.5, 0.5],
        &[0, 1],
        &[0, 1],
        2048,
        false,
    );
    for probability in player.iter().chain(&enemy) {
        assert!((probability - 0.5).abs() < 1e-12);
    }
    assert!(value.abs() < 1e-12);
    assert!(exploitability.abs() < 1e-12);
}

/// Rock-paper-scissors sits on the same uniform fixpoint; the three
/// accumulated averages round identically, so they must match bitwise.
#[test]
fn rock_paper_scissors_stays_uniform_and_symmetric() {
    #[rustfmt::skip]
    let payoff = [
         0.0, -1.0,  1.0,
         1.0,  0.0, -1.0,
        -1.0,  1.0,  0.0,
    ];
    let uniform = [1.0 / 3.0; 3];
    let legal = [0, 1, 2];
    let (player, enemy, value, exploitability) =
        solve_zero_sum_regret(&payoff, 3, &uniform, &uniform, &legal, &legal, 2048, false);
    assert_eq!(player[0].to_bits(), player[1].to_bits());
    assert_eq!(player[1].to_bits(), player[2].to_bits());
    assert_eq!(player, enemy);
    for probability in player {
        assert!((probability - 1.0 / 3.0).abs() < 1e-9);
    }
    assert!(value.abs() < 1e-9);
    assert!(exploitability.abs() < 1e-9);
    assert!(exploitability >= -1e-9);
}

/// Strictly dominant actions for both sides: RM+ locks onto the pure
/// equilibrium after one iteration, so only the first iterate's prior
/// mass dilutes the average.
#[test]
fn dominant_actions_absorb_the_average_strategy() {
    // Player row 0 dominates row 1; enemy column 1 dominates column 0.
    let payoff = [2.0, 1.0, 0.0, -1.0];
    let (player, enemy, value, exploitability) = solve_zero_sum_regret(
        &payoff,
        2,
        &[0.5, 0.5],
        &[0.5, 0.5],
        &[0, 1],
        &[0, 1],
        2048,
        false,
    );
    assert!(player[0] > 0.99);
    assert!(enemy[1] > 0.99);
    assert!((value - 1.0).abs() < 0.05);
    assert!(exploitability >= -1e-9);
    assert!(exploitability < 0.05);
}

/// Zero priors renormalize to uniform, both standalone and inside the
/// solver.
#[test]
fn zero_priors_fall_back_to_uniform() {
    assert_eq!(normalized_prior(&[0.0, 0.0, 0.0, 0.0], &[1, 3]), [0.5, 0.5]);
    let renormalized = normalized_prior(&[0.2, 0.1, 0.1, 0.6], &[1, 3]);
    assert!((renormalized[0] - 1.0 / 7.0).abs() < 1e-12);
    assert!((renormalized[1] - 6.0 / 7.0).abs() < 1e-12);

    let payoff = [1.0, -1.0, -1.0, 1.0];
    let (player, _, value, _) = solve_zero_sum_regret(
        &payoff,
        2,
        &[0.0, 0.0],
        &[0.0, 0.0],
        &[0, 1],
        &[0, 1],
        512,
        false,
    );
    assert!((player[0] - 0.5).abs() < 1e-12);
    assert!(value.abs() < 1e-12);
}

/// Legal subsets: outputs are distributions over the legal actions and
/// exactly zero elsewhere, and a single-action side stays pure.
#[test]
fn policies_are_distributions_over_the_legal_subset() {
    #[rustfmt::skip]
    let payoff = [
        0.0, 0.3, 0.0,
        9.0, 9.0, 9.0, // illegal player row: must not leak into the solve
        0.0, -0.2, 0.0,
    ];
    let uniform = [1.0 / 3.0; 3];
    let (player, enemy, value, exploitability) =
        solve_zero_sum_regret(&payoff, 3, &uniform, &uniform, &[0, 2], &[1], 2048, false);
    assert_eq!(player[1], 0.0);
    assert_eq!(enemy, [0.0, 1.0, 0.0]);
    let player_total: f64 = player.iter().sum();
    assert!((player_total - 1.0).abs() < 1e-9);
    assert!(player[0] > 0.99);
    assert!((value - 0.3).abs() < 0.01);
    assert!(exploitability >= -1e-9);
}

#[test]
#[should_panic(expected = "regret iterations must be positive")]
fn zero_iterations_panics() {
    solve_zero_sum_regret(
        &[1.0, -1.0, -1.0, 1.0],
        2,
        &[0.5, 0.5],
        &[0.5, 0.5],
        &[0, 1],
        &[0, 1],
        0,
        false,
    );
}

#[test]
#[should_panic(expected = "at least one legal action")]
fn empty_legal_set_panics() {
    solve_zero_sum_regret(
        &[1.0, -1.0, -1.0, 1.0],
        2,
        &[0.5, 0.5],
        &[0.5, 0.5],
        &[],
        &[0, 1],
        16,
        false,
    );
}

/// The solver is a pure function of its inputs: repeat calls must agree
/// bit for bit, pinning the accumulation order.
#[test]
fn repeat_solves_are_bitwise_identical() {
    let payoff = [0.3, -0.8, -0.5, 0.9];
    let solve = || {
        solve_zero_sum_regret(
            &payoff,
            2,
            &[0.7, 0.3],
            &[0.4, 0.6],
            &[0, 1],
            &[0, 1],
            2048,
            false,
        )
    };
    let (player_a, enemy_a, value_a, exploitability_a) = solve();
    let (player_b, enemy_b, value_b, exploitability_b) = solve();
    let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&player_a), bits(&player_b));
    assert_eq!(bits(&enemy_a), bits(&enemy_b));
    assert_eq!(value_a.to_bits(), value_b.to_bits());
    assert_eq!(exploitability_a.to_bits(), exploitability_b.to_bits());
    // The mixed equilibrium of this game is interior; RM+ at 2048
    // iterations should be close to it.
    assert!(exploitability_a >= -1e-9);
    assert!(exploitability_a < 0.05);
    assert!(player_a[0] > 0.0 && player_a[1] > 0.0);
}

// --- Tolerance laws (hegel) -----------------------------------------------

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn draw_matrix(tc: &TestCase, action_count: usize) -> Vec<f64> {
    tc.draw(
        gs::vecs(gs::floats::<f64>().min_value(-1.0).max_value(1.0))
            .min_size(action_count * action_count)
            .max_size(action_count * action_count),
    )
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

/// A `None` tolerance is the plain solver bit for bit, with the full
/// iteration count reported as performed: the convergence checks are
/// pure reads that must not perturb a single intermediate.
#[hegel::test(test_cases = 60)]
fn tolerance_none_matches_the_plain_solver_bitwise(tc: TestCase) {
    let action_count: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
    let payoff = draw_matrix(&tc, action_count);
    let player_legal = draw_legal(&tc, action_count);
    let enemy_legal = draw_legal(&tc, action_count);
    let player_priors = draw_priors(&tc, action_count);
    let enemy_priors = draw_priors(&tc, action_count);
    let iterations: u32 = tc.draw(gs::integers::<u32>().min_value(1).max_value(320));
    let cfr_plus: bool = tc.draw(gs::booleans());

    let (player, enemy, value, exploitability) = solve_zero_sum_regret(
        &payoff,
        action_count,
        &player_priors,
        &enemy_priors,
        &player_legal,
        &enemy_legal,
        iterations,
        cfr_plus,
    );
    let (tolerant_player, tolerant_enemy, tolerant_value, tolerant_exploitability, performed) =
        solve_zero_sum_regret_with_tolerance(
            &payoff,
            action_count,
            &player_priors,
            &enemy_priors,
            &player_legal,
            &enemy_legal,
            iterations,
            cfr_plus,
            None,
        );
    assert_eq!(bits(&player), bits(&tolerant_player));
    assert_eq!(bits(&enemy), bits(&tolerant_enemy));
    assert_eq!(value.to_bits(), tolerant_value.to_bits());
    assert_eq!(exploitability.to_bits(), tolerant_exploitability.to_bits());
    assert_eq!(performed, iterations);
}

/// A tolerance stop is a bitwise truncation of the fixed-iteration
/// solve: the outputs equal the plain solver run for exactly
/// `performed` iterations, and `performed` is the first
/// interval-aligned checkpoint whose time-average exploitability meets
/// the tolerance — every earlier checkpoint's truncation still exceeds
/// it.
#[hegel::test(test_cases = 60)]
fn tolerance_stops_are_bitwise_truncations_at_the_first_converged_checkpoint(tc: TestCase) {
    let action_count: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
    let payoff = draw_matrix(&tc, action_count);
    let player_legal = draw_legal(&tc, action_count);
    let enemy_legal = draw_legal(&tc, action_count);
    let player_priors = draw_priors(&tc, action_count);
    let enemy_priors = draw_priors(&tc, action_count);
    let iterations: u32 = tc.draw(gs::integers::<u32>().min_value(1).max_value(320));
    let cfr_plus: bool = tc.draw(gs::booleans());
    let tolerance: f64 = tc.draw(gs::floats::<f64>().min_value(1e-4).max_value(2.5));

    let (player, enemy, value, exploitability, performed) = solve_zero_sum_regret_with_tolerance(
        &payoff,
        action_count,
        &player_priors,
        &enemy_priors,
        &player_legal,
        &enemy_legal,
        iterations,
        cfr_plus,
        Some(tolerance),
    );
    assert!(performed >= 1 && performed <= iterations);
    if performed < iterations {
        assert_eq!(
            performed % EQUILIBRIUM_CHECK_INTERVAL,
            0,
            "an early stop must land on a checkpoint"
        );
        assert!(
            exploitability <= tolerance,
            "an early stop must meet the tolerance: {exploitability} vs {tolerance}"
        );
    }

    let (truncated_player, truncated_enemy, truncated_value, truncated_exploitability) =
        solve_zero_sum_regret(
            &payoff,
            action_count,
            &player_priors,
            &enemy_priors,
            &player_legal,
            &enemy_legal,
            performed,
            cfr_plus,
        );
    assert_eq!(bits(&player), bits(&truncated_player));
    assert_eq!(bits(&enemy), bits(&truncated_enemy));
    assert_eq!(value.to_bits(), truncated_value.to_bits());
    assert_eq!(exploitability.to_bits(), truncated_exploitability.to_bits());

    // Minimality: the checkpoint exploitability the stop rule reads is
    // exactly the truncated run's, so every checkpoint before the stop
    // must still have been above the tolerance.
    let mut checkpoint = EQUILIBRIUM_CHECK_INTERVAL;
    while checkpoint < performed {
        let (_, _, _, early_exploitability) = solve_zero_sum_regret(
            &payoff,
            action_count,
            &player_priors,
            &enemy_priors,
            &player_legal,
            &enemy_legal,
            checkpoint,
            cfr_plus,
        );
        assert!(
            early_exploitability > tolerance,
            "checkpoint {checkpoint} already met the tolerance but the \
             solve ran to {performed}"
        );
        checkpoint += EQUILIBRIUM_CHECK_INTERVAL;
    }
}
