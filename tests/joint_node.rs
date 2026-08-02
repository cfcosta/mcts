//! Characterization tests for tree nodes, the warm solver, and the pure
//! descent helpers, ported from the Python suite. Exact equalities are
//! asserted only where every intermediate is exactly representable.

use mcts_rs::joint::{
    argmax_first, average_policy, chance_resample_probability, expansion_pairs, legal_from_priors,
    mixed_policy, policy_entropy, sample_index, solve_node, Evaluation, JointSearchConfig,
    JointSnapshot, Outcome, SplitMix64, Tree,
};

/// A minimal non-terminal snapshot: masks only.
struct Snap {
    player_mask: u64,
    enemy_mask: u64,
}

impl JointSnapshot for Snap {
    fn id(&self) -> u64 {
        0
    }
    fn player_mask(&self) -> u64 {
        self.player_mask
    }
    fn enemy_mask(&self) -> u64 {
        self.enemy_mask
    }
    fn terminal_value(&self) -> Option<f64> {
        None
    }
}

/// A tree whose root has every action legal, the given full-length priors,
/// and the given row-major payoff already installed.
fn tree_with_root(payoff: &[f64], player_priors: &[f64], enemy_priors: &[f64]) -> Tree<Snap> {
    let n = player_priors.len();
    assert_eq!(payoff.len(), n * n);
    let mut tree = Tree::new(n);
    let mask = (1u64 << n) - 1;
    tree.make_node(
        Snap {
            player_mask: mask,
            enemy_mask: mask,
        },
        Evaluation {
            player_priors: player_priors.to_vec(),
            enemy_priors: enemy_priors.to_vec(),
            value: 0.0,
        },
        &JointSearchConfig::default(),
    );
    tree.nodes[0].payoff.copy_from_slice(payoff);
    tree
}

#[test]
fn legal_from_priors_orders_by_prior_then_index_and_caps() {
    let priors = [0.1, 0.4, 0.4, 0.1];
    assert_eq!(legal_from_priors(0b1111, &priors, 3), [1, 2, 0]);
    assert_eq!(legal_from_priors(0b1111, &priors, 13), [1, 2, 0, 3]);
    assert_eq!(legal_from_priors(0b1010, &priors, 13), [1, 3]);
    // Mask bits at or above the action count are ignored, mirroring the
    // Python range(action_count) scan.
    assert_eq!(legal_from_priors((1 << 40) | 0b01, &[0.5, 0.5], 13), [0]);
}

#[test]
fn make_node_prefills_the_legal_grid_with_the_leaf_value() {
    let mut tree = Tree::new(3);
    tree.make_node(
        Snap {
            player_mask: 0b101,
            enemy_mask: 0b010,
        },
        Evaluation {
            player_priors: vec![0.2, 0.9, 0.3],
            enemy_priors: vec![0.1, 0.5, 0.4],
            value: 0.25,
        },
        &JointSearchConfig::default(),
    );
    let node = tree.root();
    assert_eq!(node.player_legal, [2, 0]);
    assert_eq!(node.enemy_legal, [1]);
    assert_eq!(node.leaf_value, 0.25);
    for player in 0..3 {
        for enemy in 0..3 {
            let expected = if (player == 2 || player == 0) && enemy == 1 {
                0.25
            } else {
                0.0
            };
            assert_eq!(node.payoff_at(player, enemy), expected);
            assert_eq!(node.count_at(player, enemy), 0);
            assert!(node.outcomes_at(player, enemy).is_empty());
        }
    }
    assert!(node.player_policy.is_empty() && node.enemy_policy.is_empty());
    assert!(!node.expanded);
    assert_eq!((node.visits, node.solve_count), (0, 0));
    assert_eq!(node.player_regrets, vec![0.0; 3]);
    assert_eq!(node.player_strategy_sum, vec![0.0; 3]);
}

#[test]
#[should_panic(expected = "at least one legal action per side")]
fn make_node_panics_when_a_side_has_no_legal_action() {
    let mut tree = Tree::new(2);
    tree.make_node(
        Snap {
            player_mask: 0b11,
            enemy_mask: 0,
        },
        Evaluation {
            player_priors: vec![0.5, 0.5],
            enemy_priors: vec![0.5, 0.5],
            value: 0.0,
        },
        &JointSearchConfig::default(),
    );
}

#[test]
#[should_panic(expected = "within 1..=64")]
fn zero_action_trees_panic() {
    Tree::<Snap>::new(0);
}

#[test]
fn record_value_maintains_the_running_mean() {
    let mut tree = tree_with_root(&[0.7; 4], &[0.5, 0.5], &[0.5, 0.5]);
    let node = tree.node_mut(0);
    // The first record replaces the prefill entirely: count was 0.
    node.record_value(0, 1, 1.0);
    assert_eq!(node.payoff_at(0, 1), 1.0);
    assert_eq!(node.count_at(0, 1), 1);
    node.record_value(0, 1, 0.0);
    assert_eq!(node.payoff_at(0, 1), 0.5);
    node.record_value(0, 1, 0.25);
    assert_eq!(node.payoff_at(0, 1), (0.5 * 2.0 + 0.25) / 3.0);
    assert_eq!(node.count_at(0, 1), 3);
    // Other cells are untouched.
    assert_eq!(node.payoff_at(1, 0), 0.7);
    assert_eq!(node.count_at(1, 0), 0);
}

#[test]
fn outcome_cells_start_empty_and_accumulate() {
    let mut tree = tree_with_root(&[0.0; 4], &[0.5, 0.5], &[0.5, 0.5]);
    let node = tree.node_mut(0);
    node.push_outcome(
        1,
        0,
        Outcome {
            snapshot: Snap {
                player_mask: 0b11,
                enemy_mask: 0b11,
            },
            tactical_delta: 0.0,
            leaf_value: 0.5,
        },
    );
    assert_eq!(node.outcomes_at(1, 0).len(), 1);
    assert!(node.outcomes_at(0, 1).is_empty());
}

/// Matching pennies with uniform priors is an exact RM+ fixpoint for the
/// warm solver too: every iterate is the prior, regrets never grow, and
/// each solve adds exactly `iterations / 2` to each strategy sum.
#[test]
fn warm_solve_on_matching_pennies_is_the_exact_fixpoint() {
    let mut tree = tree_with_root(&[1.0, -1.0, -1.0, 1.0], &[0.5, 0.5], &[0.5, 0.5]);
    let node = tree.node_mut(0);
    solve_node(node, 16, false);
    assert_eq!(node.player_policy, [0.5, 0.5]);
    assert_eq!(node.enemy_policy, [0.5, 0.5]);
    assert_eq!(node.root_value, 0.0);
    assert_eq!(node.exploitability, 0.0);
    assert_eq!(node.solve_count, 16);
    assert_eq!(node.player_strategy_sum, [8.0, 8.0]);
    assert_eq!(node.player_regrets, [0.0, 0.0]);
    solve_node(node, 16, false);
    assert_eq!(node.solve_count, 32);
    assert_eq!(node.player_strategy_sum, [16.0, 16.0]);
}

/// A dominant-strategy game locks onto the pure equilibrium at iteration
/// 2, so the last-iterate policy is exactly pure while the running
/// average keeps the first iteration's uniform prior — pinning that the
/// node policy is the last iterate, not the average.
#[test]
fn warm_solve_installs_the_last_iterate_not_the_average() {
    // Player row 0 dominates; enemy column 1 dominates.
    let mut tree = tree_with_root(&[2.0, 1.0, 0.0, -1.0], &[0.5, 0.5], &[0.5, 0.5]);
    let node = tree.node_mut(0);
    solve_node(node, 16, false);
    assert_eq!(node.player_policy, [1.0, 0.0]);
    assert_eq!(node.enemy_policy, [0.0, 1.0]);
    assert_eq!(node.root_value, 1.0);
    assert_eq!(node.exploitability, 0.0);
    // 15 pure iterations plus the uniform first iterate.
    assert_eq!(node.player_strategy_sum, [15.5, 0.5]);
    let average = average_policy(
        &node.player_strategy_sum,
        node.solve_count,
        &node.player_policy,
    );
    assert_eq!(average, [0.96875, 0.03125]);
}

/// Warm regrets never trap the solver: flipping the payoff to favor the
/// other action flips the last-iterate policy within a single update.
#[test]
fn warm_solve_moves_mass_after_a_payoff_flip() {
    let mut tree = tree_with_root(&[2.0, 1.0, 0.0, -1.0], &[0.5, 0.5], &[0.5, 0.5]);
    let node = tree.node_mut(0);
    solve_node(node, 16, false);
    assert_eq!(node.player_policy, [1.0, 0.0]);
    // Row 1 now dominates; the flip must overcome the accumulated regret.
    node.payoff.copy_from_slice(&[0.0, -1.0, 2.0, 1.0]);
    solve_node(node, 16, false);
    assert_eq!(node.player_policy, [0.0, 1.0]);
    assert_eq!(node.root_value, 1.0);
}

/// Repeat warm solves from identical node states agree bit for bit.
#[test]
fn warm_solves_are_bitwise_deterministic() {
    let build = || tree_with_root(&[0.3, -0.8, -0.5, 0.9], &[0.7, 0.3], &[0.4, 0.6]);
    let mut first = build();
    let mut second = build();
    for _ in 0..3 {
        solve_node(first.node_mut(0), 16, false);
        solve_node(second.node_mut(0), 16, false);
    }
    let (a, b) = (first.root(), second.root());
    let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a.player_policy), bits(&b.player_policy));
    assert_eq!(bits(&a.player_regrets), bits(&b.player_regrets));
    assert_eq!(bits(&a.player_strategy_sum), bits(&b.player_strategy_sum));
    assert_eq!(a.root_value.to_bits(), b.root_value.to_bits());
    assert_eq!(a.exploitability.to_bits(), b.exploitability.to_bits());
}

#[test]
fn resample_probability_decays_to_the_floor() {
    assert_eq!(chance_resample_probability(0, 0.1), 1.0);
    assert_eq!(chance_resample_probability(1, 0.1), 1.0);
    assert_eq!(chance_resample_probability(2, 0.1), 1.0 / 2.0_f64.sqrt());
    assert_eq!(chance_resample_probability(100, 0.1), 0.1);
    assert_eq!(chance_resample_probability(400, 0.1), 0.1);
}

#[test]
fn mixed_policy_blends_the_solve_with_the_prior() {
    // visits 0 with exploration 0.2: epsilon is exactly the exploration.
    let mixed = mixed_policy(&[1.0, 0.0], &[0.0, 1.0], &[0, 1], 0, 0.2);
    assert!((mixed[0] - 0.8).abs() < 1e-12);
    assert_eq!(mixed[1], 0.2);

    // visits 99 with exploration 0.1 decays to the 0.02 floor.
    let floored = mixed_policy(&[0.0, 1.0], &[1.0, 0.0], &[0, 1], 99, 0.1);
    assert_eq!(floored[0], 0.02);
    assert!((floored[1] - 0.98).abs() < 1e-12);

    // Zero prior mass over the legal set falls back to uniform shares,
    // and illegal actions stay exactly zero.
    let uniform = mixed_policy(&[0.5, 0.0, 0.5], &[0.0, 0.0, 0.0], &[2, 0], 0, 0.0);
    assert!((uniform[0] - 0.5).abs() < 1e-12);
    assert_eq!(uniform[1], 0.0);
    assert!((uniform[2] - 0.5).abs() < 1e-12);
}

#[test]
fn sample_index_walks_the_cumulative_distribution() {
    // The first draw from seed 0 is 0.8833108082136426.
    let mut rng = SplitMix64::new(0);
    assert_eq!(sample_index(&[0.9, 0.1], &mut rng), 0);
    let mut rng = SplitMix64::new(0);
    assert_eq!(sample_index(&[0.5, 0.5], &mut rng), 1);
    // A distribution that sums short of the draw falls back to the last
    // positive entry.
    let mut rng = SplitMix64::new(0);
    assert_eq!(sample_index(&[0.1, 0.2, 0.0], &mut rng), 1);
    // A leading zero can never absorb the draw.
    let mut rng = SplitMix64::new(7);
    assert_eq!(sample_index(&[0.0, 1.0], &mut rng), 1);
}

#[test]
#[should_panic(expected = "no positive mass")]
fn sampling_from_a_zero_distribution_panics() {
    let mut rng = SplitMix64::new(0);
    sample_index(&[0.0, 0.0], &mut rng);
}

#[test]
fn argmax_first_prefers_the_earlier_legal_entry_on_ties() {
    let policy = [0.3, 0.5, 0.5, 0.2];
    assert_eq!(argmax_first(&policy, &[2, 1, 0]), 2);
    assert_eq!(argmax_first(&policy, &[1, 2, 3]), 1);
    assert_eq!(argmax_first(&policy, &[3]), 3);
}

#[test]
fn expansion_pairs_covers_the_grid_or_rotates_diagonals() {
    // Full matrix: player-outer product in legal-list order.
    assert_eq!(
        expansion_pairs(&[1, 0], &[2, 3], true, 2),
        [(1, 2), (1, 3), (0, 2), (0, 3)]
    );
    // Partial: max(|P|, |E|) pairs per rotation, wrapping each side.
    let rotated = expansion_pairs(&[0, 1, 2], &[7, 8], false, 2);
    assert_eq!(rotated, [(0, 7), (1, 8), (2, 7), (0, 8), (1, 7), (2, 8)]);
    for &player in &[0, 1, 2] {
        assert!(rotated.iter().any(|&(p, _)| p == player));
    }
    for &enemy in &[7, 8] {
        assert!(rotated.iter().any(|&(_, e)| e == enemy));
    }
    // Wrapping duplicates are dropped in first-seen order.
    assert_eq!(expansion_pairs(&[0], &[5, 6], false, 2), [(0, 5), (0, 6)]);
    // Rotations are capped by the enemy side's size.
    assert_eq!(expansion_pairs(&[0, 1], &[9], false, 2), [(0, 9), (1, 9)]);
}

#[test]
fn policy_entropy_sums_over_positive_entries() {
    assert!((policy_entropy(&[0.5, 0.0, 0.5]) - 2.0_f64.ln()).abs() < 1e-12);
    assert!((policy_entropy(&[0.25; 4]) - 4.0_f64.ln()).abs() < 1e-12);
    assert_eq!(policy_entropy(&[1.0, 0.0]), 0.0);
    assert_eq!(policy_entropy(&[0.0, 0.0]), 0.0);
}

#[test]
fn average_policy_divides_by_solves_or_falls_back() {
    assert_eq!(average_policy(&[8.0, 24.0], 32, &[0.0, 0.0]), [0.25, 0.75]);
    assert_eq!(average_policy(&[8.0, 24.0], 0, &[0.1, 0.9]), [0.1, 0.9]);
}

#[test]
fn tree_ids_are_assigned_in_creation_order() {
    let mut tree = Tree::new(2);
    let evaluation = || Evaluation {
        player_priors: vec![0.5, 0.5],
        enemy_priors: vec![0.5, 0.5],
        value: 0.0,
    };
    let snap = || Snap {
        player_mask: 0b11,
        enemy_mask: 0b11,
    };
    assert_eq!(
        tree.make_node(snap(), evaluation(), &JointSearchConfig::default()),
        0
    );
    assert_eq!(
        tree.make_node(snap(), evaluation(), &JointSearchConfig::default()),
        1
    );
    tree.node_mut(1).visits = 3;
    assert_eq!(tree.node(1).visits, 3);
    assert_eq!(tree.root().visits, 0);
}
