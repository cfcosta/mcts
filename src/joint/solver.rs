//! Regret-matching+ solvers over dense payoff matrices.
//!
//! Ports `solve_zero_sum_regret` and `_normalized_prior` from the Python
//! search. Matrix products accumulate sequentially left-to-right — the
//! crate's frozen summation order — so results are deterministic here but
//! not bit-identical to NumPy's pairwise/BLAS accumulation; ported
//! characterization tests therefore assert with tolerances, never against
//! Python-produced bit patterns.

/// Renormalizes `priors` over the `legal` subset, compact and ordered as
/// `legal`; uniform when no prior mass survives.
pub fn normalized_prior(priors: &[f64], legal: &[usize]) -> Vec<f64> {
    assert!(
        !legal.is_empty(),
        "cannot normalize over an empty legal set"
    );
    let values: Vec<f64> = legal.iter().map(|&action| priors[action]).collect();
    let total: f64 = values.iter().sum();
    if total > 0.0 {
        values.into_iter().map(|value| value / total).collect()
    } else {
        vec![1.0 / legal.len() as f64; legal.len()]
    }
}

/// Approximates a zero-sum matrix equilibrium with regret matching+.
///
/// `payoff` is the full `action_count`² player-payoff matrix, row-major
/// with rows indexed by player action; the solve runs on the legal
/// submatrix. Returns `(player_policy, enemy_policy, value,
/// exploitability)` where the policies are the **time-average** strategies
/// scattered to full length, and value/exploitability are recomputed on
/// those averages (`exploitability = max(M·e) - min(pᵀ·M)`).
pub fn solve_zero_sum_regret(
    payoff: &[f64],
    action_count: usize,
    player_priors: &[f64],
    enemy_priors: &[f64],
    player_legal: &[usize],
    enemy_legal: &[usize],
    iterations: u32,
) -> (Vec<f64>, Vec<f64>, f64, f64) {
    assert!(iterations >= 1, "regret iterations must be positive");
    assert!(
        !player_legal.is_empty() && !enemy_legal.is_empty(),
        "each side needs at least one legal action"
    );
    assert_eq!(
        payoff.len(),
        action_count * action_count,
        "payoff must be an action_count x action_count matrix"
    );
    assert_eq!(player_priors.len(), action_count, "player prior length");
    assert_eq!(enemy_priors.len(), action_count, "enemy prior length");

    let player_len = player_legal.len();
    let enemy_len = enemy_legal.len();
    let mut matrix = Vec::with_capacity(player_len * enemy_len);
    for &player in player_legal {
        for &enemy in enemy_legal {
            matrix.push(payoff[player * action_count + enemy]);
        }
    }
    let player_prior = normalized_prior(player_priors, player_legal);
    let enemy_prior = normalized_prior(enemy_priors, enemy_legal);

    let mut player_regrets = vec![0.0; player_len];
    let mut enemy_regrets = vec![0.0; enemy_len];
    let mut player_sum = vec![0.0; player_len];
    let mut enemy_sum = vec![0.0; enemy_len];
    let mut player = vec![0.0; player_len];
    let mut enemy = vec![0.0; enemy_len];
    let mut row_values = vec![0.0; player_len];
    let mut column_values = vec![0.0; enemy_len];
    for _ in 0..iterations {
        regret_strategy(&player_regrets, &player_prior, &mut player);
        regret_strategy(&enemy_regrets, &enemy_prior, &mut enemy);
        for (sum, &probability) in player_sum.iter_mut().zip(&player) {
            *sum += probability;
        }
        for (sum, &probability) in enemy_sum.iter_mut().zip(&enemy) {
            *sum += probability;
        }
        mat_vec(&matrix, enemy_len, &enemy, &mut row_values);
        vec_mat(&player, &matrix, enemy_len, &mut column_values);
        let current_value = dot(&player, &row_values);
        for (regret, &row_value) in player_regrets.iter_mut().zip(&row_values) {
            *regret = (*regret + row_value - current_value).max(0.0);
        }
        for (regret, &column_value) in enemy_regrets.iter_mut().zip(&column_values) {
            *regret = (*regret + current_value - column_value).max(0.0);
        }
    }

    let player_average: Vec<f64> = player_sum
        .iter()
        .map(|sum| sum / f64::from(iterations))
        .collect();
    let enemy_average: Vec<f64> = enemy_sum
        .iter()
        .map(|sum| sum / f64::from(iterations))
        .collect();
    let mut player_policy = vec![0.0; action_count];
    let mut enemy_policy = vec![0.0; action_count];
    for (index, &action) in player_legal.iter().enumerate() {
        player_policy[action] = player_average[index];
    }
    for (index, &action) in enemy_legal.iter().enumerate() {
        enemy_policy[action] = enemy_average[index];
    }
    mat_vec(&matrix, enemy_len, &enemy_average, &mut row_values);
    vec_mat(&player_average, &matrix, enemy_len, &mut column_values);
    let value = dot(&player_average, &row_values);
    let exploitability = max_of(&row_values) - min_of(&column_values);
    (player_policy, enemy_policy, value, exploitability)
}

/// The RM+ per-iteration strategy: positive regrets normalized, falling
/// back to the prior when no regret mass exists.
fn regret_strategy(regrets: &[f64], prior: &[f64], strategy: &mut [f64]) {
    let mut total = 0.0;
    for (slot, &regret) in strategy.iter_mut().zip(regrets) {
        let positive = regret.max(0.0);
        *slot = positive;
        total += positive;
    }
    if total > 0.0 {
        for slot in strategy.iter_mut() {
            *slot /= total;
        }
    } else {
        strategy.copy_from_slice(prior);
    }
}

/// `out = matrix · vector` for a row-major matrix with `columns` columns.
fn mat_vec(matrix: &[f64], columns: usize, vector: &[f64], out: &mut [f64]) {
    for (row, slot) in matrix.chunks_exact(columns).zip(out.iter_mut()) {
        *slot = dot(row, vector);
    }
}

/// `out = vector · matrix`, accumulating rows top to bottom so every
/// output element sums in ascending row order.
fn vec_mat(vector: &[f64], matrix: &[f64], columns: usize, out: &mut [f64]) {
    out.fill(0.0);
    for (row, &weight) in matrix.chunks_exact(columns).zip(vector) {
        for (slot, &cell) in out.iter_mut().zip(row) {
            *slot += weight * cell;
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn max_of(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn min_of(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}
