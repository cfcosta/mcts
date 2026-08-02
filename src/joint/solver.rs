//! Regret-matching+ solvers over dense payoff matrices.
//!
//! The cold solver approximates a matrix equilibrium from scratch; the
//! warm solver iterates a tree node's persistent regrets in place. Matrix
//! products accumulate sequentially left-to-right — the crate's frozen
//! summation order — so every solve is bitwise reproducible.

use rand::RngCore;

use crate::joint::node::TreeNode;
use crate::joint::rng::next_f64;

/// Renormalizes `priors` over the `legal` subset, compact and ordered as
/// `legal`; uniform when no prior mass survives.
pub fn normalized_prior(priors: &[f64], legal: &[usize]) -> Vec<f64> {
    let mut prior = Vec::new();
    normalized_prior_into(priors, legal, &mut prior);
    prior
}

/// [`normalized_prior`] into a caller-held buffer: one exact allocation
/// when the buffer is too small, none once it is warm. Both branches
/// renormalize the gathered entries in place, so a mass-free prior costs
/// no extra buffer.
fn normalized_prior_into(priors: &[f64], legal: &[usize], out: &mut Vec<f64>) {
    assert!(
        !legal.is_empty(),
        "cannot normalize over an empty legal set"
    );
    refill_gather(out, priors, legal);
    let total: f64 = out.iter().sum();
    if total > 0.0 {
        for value in out.iter_mut() {
            *value /= total;
        }
    } else {
        out.fill(1.0 / legal.len() as f64);
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
///
/// With `cfr_plus` (the
/// [`cfr_plus_solves`](crate::joint::config::JointSearchConfig::cfr_plus_solves)
/// extension) the iteration body uses CFR+'s two accelerations
/// (Tammelin, arXiv:1407.5042): **alternating updates** — the enemy's
/// regrets are updated against the player strategy already refreshed
/// from this iteration's player-regret update, instead of both sides
/// updating against the stale pair — and **linear averaging** —
/// iteration `t` enters the strategy average with weight `t`, so the
/// average forgets the poor early iterates at a quadratic rate. With
/// the flag off, both sides update against the stale pair and every
/// iteration enters the average with weight 1 — the default
/// simultaneous uniform-average dynamics.
// The wide surface is deliberate: the characterization tests pin the
// solver by calling it directly with every input visible; a params
// struct would only obscure them.
#[allow(clippy::too_many_arguments)]
pub fn solve_zero_sum_regret(
    payoff: &[f64],
    action_count: usize,
    player_priors: &[f64],
    enemy_priors: &[f64],
    player_legal: &[usize],
    enemy_legal: &[usize],
    iterations: u32,
    cfr_plus: bool,
) -> (Vec<f64>, Vec<f64>, f64, f64) {
    let (player_policy, enemy_policy, value, exploitability, _) =
        solve_zero_sum_regret_with_tolerance(
            payoff,
            action_count,
            player_priors,
            enemy_priors,
            player_legal,
            enemy_legal,
            iterations,
            cfr_plus,
            None,
        );
    (player_policy, enemy_policy, value, exploitability)
}

/// Iteration interval at which a tolerance-carrying solve inspects its
/// time-average exploitability.
pub const EQUILIBRIUM_CHECK_INTERVAL: u32 = 64;

/// [`solve_zero_sum_regret`] with an optional early-termination
/// tolerance (the
/// [`equilibrium_tolerance`](crate::joint::config::JointSearchConfig::equilibrium_tolerance)
/// extension); the fifth returned element is the number of iterations
/// actually performed.
///
/// With `Some(tolerance)`, every [`EQUILIBRIUM_CHECK_INTERVAL`]-th
/// iteration — except a final one, which stops regardless — computes
/// the exploitability of the current time-average strategies, the
/// exact expression the tail evaluates, and stops the loop once it is
/// at most `tolerance`. The check reads the running sums into two
/// dedicated buffers and writes nothing the iterations read, so a
/// stopped solve returns bit for bit what a plain solve of `performed`
/// iterations returns, and a tolerance no checkpoint meets — an
/// unreachable one, or any non-finite garbage — leaves the full run
/// bitwise untouched. With `None`, no checks happen and
/// `performed == iterations`.
// See `solve_zero_sum_regret` for the surface rationale.
#[allow(clippy::too_many_arguments)]
pub fn solve_zero_sum_regret_with_tolerance(
    payoff: &[f64],
    action_count: usize,
    player_priors: &[f64],
    enemy_priors: &[f64],
    player_legal: &[usize],
    enemy_legal: &[usize],
    iterations: u32,
    cfr_plus: bool,
    tolerance: Option<f64>,
) -> (Vec<f64>, Vec<f64>, f64, f64, u32) {
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
    // Filled lazily on the first checkpoint, so a `None` tolerance or a
    // short run never pays for them.
    let mut player_average_check: Vec<f64> = Vec::new();
    let mut enemy_average_check: Vec<f64> = Vec::new();
    let mut performed = iterations;
    for iteration in 0..iterations {
        regret_strategy(&player_regrets, &player_prior, &mut player);
        regret_strategy(&enemy_regrets, &enemy_prior, &mut enemy);
        // Off, the weight is exactly 1.0 and `1.0 * x` is `x` bitwise, so
        // the uniform accumulation is preserved exactly.
        let weight = if cfr_plus {
            f64::from(iteration + 1)
        } else {
            1.0
        };
        for (sum, &probability) in player_sum.iter_mut().zip(&player) {
            *sum += weight * probability;
        }
        for (sum, &probability) in enemy_sum.iter_mut().zip(&enemy) {
            *sum += weight * probability;
        }
        mat_vec(&matrix, enemy_len, &enemy, &mut row_values);
        let current_value = dot(&player, &row_values);
        for (regret, &row_value) in player_regrets.iter_mut().zip(&row_values) {
            *regret = (*regret + row_value - current_value).max(0.0);
        }
        // Alternation: the enemy responds to the player strategy induced
        // by the regrets just updated above, not the stale iterate. Off,
        // the refresh is skipped and `column_values`/`enemy_value` are
        // bitwise what the simultaneous body computes — the player-regret
        // update above writes `player_regrets`, never `player`, so
        // moving `vec_mat` below it changes nothing.
        if cfr_plus {
            regret_strategy(&player_regrets, &player_prior, &mut player);
        }
        vec_mat(&player, &matrix, enemy_len, &mut column_values);
        let enemy_value = if cfr_plus {
            dot(&enemy, &column_values)
        } else {
            current_value
        };
        for (regret, &column_value) in enemy_regrets.iter_mut().zip(&column_values) {
            *regret = (*regret + enemy_value - column_value).max(0.0);
        }
        if let Some(tolerance) = tolerance {
            let solved = iteration + 1;
            if solved % EQUILIBRIUM_CHECK_INTERVAL == 0 && solved < iterations {
                let weight_total = strategy_weight_total(cfr_plus, solved);
                refill_average(&mut player_average_check, &player_sum, weight_total);
                refill_average(&mut enemy_average_check, &enemy_sum, weight_total);
                // Scratching `row_values`/`column_values` is invisible
                // to the iterations: the next `mat_vec` overwrites every
                // slot before reading it and `vec_mat` zero-fills first,
                // as does the tail below.
                mat_vec(&matrix, enemy_len, &enemy_average_check, &mut row_values);
                vec_mat(
                    &player_average_check,
                    &matrix,
                    enemy_len,
                    &mut column_values,
                );
                if max_of(&row_values) - min_of(&column_values) <= tolerance {
                    performed = solved;
                    break;
                }
            }
        }
    }

    let weight_total = strategy_weight_total(cfr_plus, performed);
    let player_average: Vec<f64> = player_sum.iter().map(|sum| sum / weight_total).collect();
    let enemy_average: Vec<f64> = enemy_sum.iter().map(|sum| sum / weight_total).collect();
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
    (
        player_policy,
        enemy_policy,
        value,
        exploitability,
        performed,
    )
}

/// Warm-started RM+ on a node's legal submatrix.
///
/// Differs from [`solve_zero_sum_regret`] on every output: regrets carry
/// over between calls through the node (strategy sums are fresh per call
/// and accumulated into `strategy_sum`), the installed policy is the
/// **last iterate** rather than the time average, and value/exploitability
/// are computed on that last iterate. The iteration body itself is
/// identical to the cold solver's.
///
/// With `average_policies` (the
/// [`average_strategy_policies`](crate::joint::config::JointSearchConfig::average_strategy_policies)
/// extension) the installed policy is instead the cumulative time
/// average `strategy_sum / strategy_weight_total(..)` over every solve
/// so far — including this call's — with value and exploitability
/// recomputed on the averages exactly as the cold solver does. On a
/// node's first solve this reproduces [`solve_zero_sum_regret`]
/// bitwise. The solver state written back (regrets, sums, count) is
/// identical in both modes; only the installed outputs differ.
///
/// With `cfr_plus` (the
/// [`cfr_plus_solves`](crate::joint::config::JointSearchConfig::cfr_plus_solves)
/// extension) the iteration body alternates the regret updates and
/// weighs strategies linearly, exactly as in the cold solver — see
/// [`solve_zero_sum_regret`]. The linear weights continue **globally**
/// across warm batches: a node at `solve_count` S entering a batch
/// weighs its next iterations `S+1, S+2, …`, so batched solves
/// accumulate the same weighted sums one long solve would.
///
/// This is [`solve_node_with_scratch`] over a fresh [`SolveScratch`];
/// callers solving many nodes should hold a scratch and call the
/// threaded form directly.
pub fn solve_node<S>(
    node: &mut TreeNode<S>,
    iterations: u32,
    average_policies: bool,
    cfr_plus: bool,
) {
    solve_node_with_scratch(
        node,
        iterations,
        average_policies,
        cfr_plus,
        &mut SolveScratch::default(),
    );
}

/// Reusable working memory for [`solve_node_with_scratch`]: the eleven
/// legal-shaped buffers a node solve fills and discards. A fresh scratch
/// is empty; each buffer grows exactly to the largest shape it has
/// served and is then rewritten in place, so a scratch threaded through
/// many solves makes every solve after the first allocation-free. The
/// contents carry no information between solves — every solve overwrites
/// every buffer before reading it.
#[derive(Debug, Default)]
pub struct SolveScratch {
    matrix: Vec<f64>,
    player_regrets: Vec<f64>,
    enemy_regrets: Vec<f64>,
    player_prior: Vec<f64>,
    enemy_prior: Vec<f64>,
    player_sum: Vec<f64>,
    enemy_sum: Vec<f64>,
    player: Vec<f64>,
    enemy: Vec<f64>,
    row_values: Vec<f64>,
    column_values: Vec<f64>,
}

/// [`solve_node`] over caller-held working memory: bitwise the same
/// solve, with the intermediates borrowed from `scratch` instead of
/// freshly allocated, and the node's installed policies rewritten in
/// place after their first install. The search engine threads one
/// scratch through every solve it performs, which reduces a deep
/// search's per-solve allocator traffic to nothing.
pub fn solve_node_with_scratch<S>(
    node: &mut TreeNode<S>,
    iterations: u32,
    average_policies: bool,
    cfr_plus: bool,
    scratch: &mut SolveScratch,
) {
    assert!(iterations >= 1, "regret iterations must be positive");
    let action_count = node.action_count();
    let player_len = node.player_legal.len();
    let enemy_len = node.enemy_legal.len();
    assert!(
        player_len > 0 && enemy_len > 0,
        "each side needs at least one legal action"
    );

    let SolveScratch {
        matrix,
        player_regrets,
        enemy_regrets,
        player_prior,
        enemy_prior,
        player_sum,
        enemy_sum,
        player,
        enemy,
        row_values,
        column_values,
    } = scratch;
    matrix.clear();
    matrix.reserve_exact(player_len * enemy_len);
    for &player_action in &node.player_legal {
        for &enemy_action in &node.enemy_legal {
            matrix.push(node.payoff[player_action * action_count + enemy_action]);
        }
    }
    refill_gather(player_regrets, &node.player_regrets, &node.player_legal);
    refill_gather(enemy_regrets, &node.enemy_regrets, &node.enemy_legal);
    normalized_prior_into(&node.player_priors, &node.player_legal, player_prior);
    normalized_prior_into(&node.enemy_priors, &node.enemy_legal, enemy_prior);
    refill_zeroed(player_sum, player_len);
    refill_zeroed(enemy_sum, enemy_len);
    // The iterates start as the priors; with at least one iteration
    // (asserted above) they are overwritten before any use, so this only
    // shapes the buffers.
    refill_copy(player, player_prior);
    refill_copy(enemy, enemy_prior);
    refill_zeroed(row_values, player_len);
    refill_zeroed(column_values, enemy_len);
    // Linear weights continue where the last batch stopped; on a fresh
    // node the base is 0.0 and `0.0 + t` is `t` bitwise, which is what
    // makes the first average-mode solve match the cold solver exactly.
    let weight_base = f64::from(node.solve_count);
    for iteration in 0..iterations {
        regret_strategy(player_regrets, player_prior, player);
        regret_strategy(enemy_regrets, enemy_prior, enemy);
        let weight = if cfr_plus {
            weight_base + f64::from(iteration + 1)
        } else {
            1.0
        };
        for (sum, &probability) in player_sum.iter_mut().zip(player.iter()) {
            *sum += weight * probability;
        }
        for (sum, &probability) in enemy_sum.iter_mut().zip(enemy.iter()) {
            *sum += weight * probability;
        }
        mat_vec(matrix, enemy_len, enemy, row_values);
        let current_value = dot(player, row_values);
        for (regret, &row_value) in player_regrets.iter_mut().zip(row_values.iter()) {
            *regret = (*regret + row_value - current_value).max(0.0);
        }
        if cfr_plus {
            regret_strategy(player_regrets, player_prior, player);
        }
        vec_mat(player, matrix, enemy_len, column_values);
        let enemy_value = if cfr_plus {
            dot(enemy, column_values)
        } else {
            current_value
        };
        for (regret, &column_value) in enemy_regrets.iter_mut().zip(column_values.iter()) {
            *regret = (*regret + enemy_value - column_value).max(0.0);
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
    if average_policies {
        // Replace the last iterates with the cumulative averages; the
        // shared tail below then installs and evaluates the averages
        // exactly as it would the iterates.
        let weight_total = strategy_weight_total(cfr_plus, node.solve_count);
        for (index, &action) in node.player_legal.iter().enumerate() {
            player[index] = node.player_strategy_sum[action] / weight_total;
        }
        for (index, &action) in node.enemy_legal.iter().enumerate() {
            enemy[index] = node.enemy_strategy_sum[action] / weight_total;
        }
    }
    scatter_policy(
        &mut node.player_policy,
        action_count,
        &node.player_legal,
        player,
    );
    scatter_policy(
        &mut node.enemy_policy,
        action_count,
        &node.enemy_legal,
        enemy,
    );
    mat_vec(matrix, enemy_len, enemy, row_values);
    node.root_value = dot(player, row_values);
    vec_mat(player, matrix, enemy_len, column_values);
    node.exploitability = max_of(row_values) - min_of(column_values);
}

/// Draws an index from a probability vector: cumulative scan returning
/// the first index whose running total reaches the draw, falling back to
/// the **last** index with positive mass when rounding leaves the total
/// short. Panics when no entry is positive.
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

/// The epsilon-mixed descent policy: the node policy blended with the
/// renormalized prior over the legal set, with a visit-decayed epsilon
/// floored at 0.02. Zero outside `legal`.
pub fn mixed_policy(
    policy: &[f64],
    priors: &[f64],
    legal: &[usize],
    visits: u32,
    exploration: f64,
) -> Vec<f64> {
    let mut result = Vec::new();
    mixed_policy_into(policy, priors, legal, visits, exploration, &mut result);
    result
}

/// [`mixed_policy`] into a caller-held buffer: one exact allocation when
/// the buffer is too small, none once it is warm. The descent reuses two
/// engine-held buffers this way, one per side, on every step.
pub fn mixed_policy_into(
    policy: &[f64],
    priors: &[f64],
    legal: &[usize],
    visits: u32,
    exploration: f64,
    out: &mut Vec<f64>,
) {
    assert_eq!(
        policy.len(),
        priors.len(),
        "policy and priors must cover the same actions"
    );
    assert!(!legal.is_empty(), "cannot mix over an empty legal set");
    let epsilon = (exploration / f64::from(visits + 1).sqrt()).max(0.02);
    let prior_total: f64 = legal.iter().map(|&action| priors[action]).sum();
    refill_zeroed(out, priors.len());
    for &action in legal {
        let exploration_share = if prior_total > 0.0 {
            priors[action] / prior_total
        } else {
            1.0 / legal.len() as f64
        };
        out[action] = (1.0 - epsilon) * policy[action] + epsilon * exploration_share;
    }
}

/// Probability of sampling a fresh chance outcome for a pair with
/// `evidence` outcomes already recorded: certain until the first outcome
/// exists, then `1/√evidence` decaying to the configured floor.
pub fn chance_resample_probability(evidence: u32, floor: f64) -> f64 {
    if evidence < 1 {
        return 1.0;
    }
    floor.max(1.0 / f64::from(evidence).sqrt())
}

/// Joint action pairs to expand. Full: the whole legal×legal grid,
/// player-outer, in legal-list order. Partial: diagonal
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
/// maximum policy mass, scanning `legal` in order. The first maximum
/// wins — and since legal lists are prior-descending, ties break toward
/// the higher prior.
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

/// Natural-log entropy over the positive entries.
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

/// The total weight the strategy average accumulated over `solve_count`
/// RM+ iterations: the plain count under the uniform scheme, the
/// triangular number `S(S+1)/2` under CFR+'s linear weights (iteration
/// `t` weighs `t`). Exact in f64 for every count below 2²⁶; solver,
/// search, and the test-side invariant checker all normalize strategy
/// sums through this one helper.
pub fn strategy_weight_total(cfr_plus: bool, solve_count: u32) -> f64 {
    let count = f64::from(solve_count);
    if cfr_plus {
        count * (count + 1.0) / 2.0
    } else {
        count
    }
}

/// The time-average policy of a warm node: `strategy_sum /
/// total_weight`, or the fallback (the node's last-iterate policy) when
/// no weight has accumulated. `total_weight` is
/// [`strategy_weight_total`] of the node's solve count.
pub fn average_policy(strategy_sum: &[f64], total_weight: f64, fallback: &[f64]) -> Vec<f64> {
    let mut result = Vec::new();
    average_policy_into(strategy_sum, total_weight, fallback, &mut result);
    result
}

/// [`average_policy`] into a caller-held buffer: one exact allocation
/// when the buffer is too small, none once it is warm. The root driver's
/// convergence tracker reuses two buffers this way across every learned
/// simulation.
pub fn average_policy_into(
    strategy_sum: &[f64],
    total_weight: f64,
    fallback: &[f64],
    out: &mut Vec<f64>,
) {
    if total_weight <= 0.0 {
        refill_copy(out, fallback);
        return;
    }
    out.clear();
    out.reserve_exact(strategy_sum.len());
    out.extend(strategy_sum.iter().map(|value| value / total_weight));
}

/// Refills `out` with the entries of `values` at the `legal` indices:
/// one exact allocation when the buffer's capacity is too small, none
/// once it is warm.
fn refill_gather(out: &mut Vec<f64>, values: &[f64], legal: &[usize]) {
    out.clear();
    out.reserve_exact(legal.len());
    out.extend(legal.iter().map(|&action| values[action]));
}

/// Refills `out` with `len` zeros, allocating like [`refill_gather`].
fn refill_zeroed(out: &mut Vec<f64>, len: usize) {
    out.clear();
    out.reserve_exact(len);
    out.resize(len, 0.0);
}

/// Refills `out` with a copy of `values`, allocating like
/// [`refill_gather`].
fn refill_copy(out: &mut Vec<f64>, values: &[f64]) {
    out.clear();
    out.reserve_exact(values.len());
    out.extend_from_slice(values);
}

/// Refills `out` with `sums / weight_total`, allocating like
/// [`refill_gather`].
fn refill_average(out: &mut Vec<f64>, sums: &[f64], weight_total: f64) {
    out.clear();
    out.reserve_exact(sums.len());
    out.extend(sums.iter().map(|sum| sum / weight_total));
}

/// Scatters a compact legal-ordered `strategy` into the full-length
/// `policy`, zero elsewhere. The first install allocates the policy
/// exactly; every later install rewrites it in place.
fn scatter_policy(policy: &mut Vec<f64>, action_count: usize, legal: &[usize], strategy: &[f64]) {
    if policy.len() == action_count {
        policy.fill(0.0);
    } else {
        *policy = vec![0.0; action_count];
    }
    for (index, &action) in legal.iter().enumerate() {
        policy[action] = strategy[index];
    }
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
