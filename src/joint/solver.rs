//! Regret-matching+ solvers over dense payoff matrices.
//!
//! Ports `solve_zero_sum_regret` and `_normalized_prior` from the Python
//! search. Matrix products accumulate sequentially left-to-right — the
//! crate's frozen summation order — so results are deterministic here but
//! not bit-identical to NumPy's pairwise/BLAS accumulation; ported
//! characterization tests therefore assert with tolerances, never against
//! Python-produced bit patterns.

use rand::RngCore;

use crate::joint::node::TreeNode;
use crate::joint::rng::next_f64;

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

/// Warm-started RM+ on a node's legal submatrix (`_solve_node`).
///
/// Differs from [`solve_zero_sum_regret`] on every output: regrets carry
/// over between calls through the node (strategy sums are fresh per call
/// and accumulated into `strategy_sum`), the installed policy is the
/// **last iterate** rather than the time average, and value/exploitability
/// are computed on that last iterate. The iteration body itself is
/// identical to the cold solver's.
pub fn solve_node<S>(node: &mut TreeNode<S>, iterations: u32) {
    assert!(iterations >= 1, "regret iterations must be positive");
    let action_count = node.action_count();
    let player_len = node.player_legal.len();
    let enemy_len = node.enemy_legal.len();
    assert!(
        player_len > 0 && enemy_len > 0,
        "each side needs at least one legal action"
    );

    let mut matrix = Vec::with_capacity(player_len * enemy_len);
    for &player in &node.player_legal {
        for &enemy in &node.enemy_legal {
            matrix.push(node.payoff[player * action_count + enemy]);
        }
    }
    let mut player_regrets: Vec<f64> = node
        .player_legal
        .iter()
        .map(|&action| node.player_regrets[action])
        .collect();
    let mut enemy_regrets: Vec<f64> = node
        .enemy_legal
        .iter()
        .map(|&action| node.enemy_regrets[action])
        .collect();
    let player_prior = normalized_prior(&node.player_priors, &node.player_legal);
    let enemy_prior = normalized_prior(&node.enemy_priors, &node.enemy_legal);
    let mut player_sum = vec![0.0; player_len];
    let mut enemy_sum = vec![0.0; enemy_len];
    // Python initializes the iterates to the priors before the loop; with
    // at least one iteration (asserted above) they are overwritten before
    // any use, but the mirror keeps the ports diffable line by line.
    let mut player = player_prior.clone();
    let mut enemy = enemy_prior.clone();
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

    for (index, &action) in node.player_legal.iter().enumerate() {
        node.player_regrets[action] = player_regrets[index];
        node.player_strategy_sum[action] += player_sum[index];
    }
    for (index, &action) in node.enemy_legal.iter().enumerate() {
        node.enemy_regrets[action] = enemy_regrets[index];
        node.enemy_strategy_sum[action] += enemy_sum[index];
    }
    node.solve_count += iterations;
    let mut player_policy = vec![0.0; action_count];
    let mut enemy_policy = vec![0.0; action_count];
    for (index, &action) in node.player_legal.iter().enumerate() {
        player_policy[action] = player[index];
    }
    for (index, &action) in node.enemy_legal.iter().enumerate() {
        enemy_policy[action] = enemy[index];
    }
    node.player_policy = player_policy;
    node.enemy_policy = enemy_policy;
    mat_vec(&matrix, enemy_len, &enemy, &mut row_values);
    node.root_value = dot(&player, &row_values);
    vec_mat(&player, &matrix, enemy_len, &mut column_values);
    node.exploitability = max_of(&row_values) - min_of(&column_values);
}

/// Draws an index from a probability vector (`_sample`): cumulative scan
/// returning the first index whose running total reaches the draw, falling
/// back to the **last** index with positive mass when rounding leaves the
/// total short. Panics when no entry is positive, as Python's `max` over
/// an empty generator would.
pub fn sample_index<R: RngCore + ?Sized>(probabilities: &[f64], rng: &mut R) -> usize {
    let threshold = next_f64(rng);
    let mut cumulative = 0.0;
    for (action, &probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if threshold <= cumulative {
            return action;
        }
    }
    probabilities
        .iter()
        .rposition(|&probability| probability > 0.0)
        .expect("cannot sample from a distribution with no positive mass")
}

/// The epsilon-mixed descent policy (`_mixed_policy`): the node policy
/// blended with the renormalized prior over the legal set, with a
/// visit-decayed epsilon floored at 0.02. Zero outside `legal`.
pub fn mixed_policy(
    policy: &[f64],
    priors: &[f64],
    legal: &[usize],
    visits: u32,
    exploration: f64,
) -> Vec<f64> {
    assert_eq!(
        policy.len(),
        priors.len(),
        "policy and priors must cover the same actions"
    );
    assert!(!legal.is_empty(), "cannot mix over an empty legal set");
    let epsilon = (exploration / f64::from(visits + 1).sqrt()).max(0.02);
    let prior_total: f64 = legal.iter().map(|&action| priors[action]).sum();
    let mut result = vec![0.0; priors.len()];
    for &action in legal {
        let exploration_share = if prior_total > 0.0 {
            priors[action] / prior_total
        } else {
            1.0 / legal.len() as f64
        };
        result[action] = (1.0 - epsilon) * policy[action] + epsilon * exploration_share;
    }
    result
}

/// Probability of sampling a fresh chance outcome for a pair with
/// `evidence` outcomes already recorded (`_chance_resample_probability`):
/// certain until the first outcome exists, then `1/√evidence` decaying to
/// the configured floor.
pub fn chance_resample_probability(evidence: u32, floor: f64) -> f64 {
    if evidence < 1 {
        return 1.0;
    }
    floor.max(1.0 / f64::from(evidence).sqrt())
}

/// Joint action pairs to expand (`_expansion_pairs`). Full: the whole
/// legal×legal grid, player-outer, in legal-list order. Partial: diagonal
/// rotations — `max(|P|, |E|)` pairs per rotation, wrapping each side
/// independently — deduplicated in first-seen order so every player row
/// and enemy column is covered without the quadratic grid.
pub fn expansion_pairs(
    player_legal: &[usize],
    enemy_legal: &[usize],
    full_matrix: bool,
    rotations: usize,
) -> Vec<(usize, usize)> {
    assert!(
        !player_legal.is_empty() && !enemy_legal.is_empty(),
        "each side needs at least one legal action"
    );
    if full_matrix {
        let mut pairs = Vec::with_capacity(player_legal.len() * enemy_legal.len());
        for &player in player_legal {
            for &enemy in enemy_legal {
                pairs.push((player, enemy));
            }
        }
        return pairs;
    }
    let pair_count = player_legal.len().max(enemy_legal.len());
    let rotations = rotations.min(enemy_legal.len());
    let mut pairs = Vec::new();
    for rotation in 0..rotations {
        for index in 0..pair_count {
            let pair = (
                player_legal[index % player_legal.len()],
                enemy_legal[(index + rotation) % enemy_legal.len()],
            );
            if !pairs.contains(&pair) {
                pairs.push(pair);
            }
        }
    }
    pairs
}

/// The deterministic action choice: the first legal action achieving the
/// maximum policy mass, scanning `legal` in order. Mirrors Python's
/// `max(legal, key=...)`, which keeps the first maximum — and since legal
/// lists are prior-descending, ties break toward the higher prior.
pub fn argmax_first(policy: &[f64], legal: &[usize]) -> usize {
    assert!(!legal.is_empty(), "cannot take an argmax over no actions");
    let mut best = legal[0];
    let mut best_mass = policy[best];
    for &action in &legal[1..] {
        if policy[action] > best_mass {
            best = action;
            best_mass = policy[action];
        }
    }
    best
}

/// Natural-log entropy over the positive entries (`_policy_entropy`).
pub fn policy_entropy(policy: &[f64]) -> f64 {
    -policy
        .iter()
        .filter(|&&probability| probability != 0.0)
        .map(|&probability| {
            debug_assert!(probability > 0.0, "policies must be non-negative");
            probability * probability.ln()
        })
        .sum::<f64>()
}

/// The time-average policy of a warm node (`_average_policy`):
/// `strategy_sum / solves`, or the fallback (the node's last-iterate
/// policy) before any solve has run.
pub fn average_policy(strategy_sum: &[f64], solves: u32, fallback: &[f64]) -> Vec<f64> {
    if solves == 0 {
        return fallback.to_vec();
    }
    strategy_sum
        .iter()
        .map(|value| value / f64::from(solves))
        .collect()
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
