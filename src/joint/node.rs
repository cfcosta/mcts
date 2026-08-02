//! The joint search tree: matrix-bearing nodes in a flat arena.
//!
//! Nodes live in a plain `Vec` indexed by [`NodeId`] (root = 0) rather
//! than a bump arena: a joint tree holds tens of heavy matrix-bearing
//! nodes, not the tens of thousands of tiny nodes the UCT arena is built
//! for. All per-cell state is flattened to row-major `action_count`²
//! vectors, and a cell's outcome list being empty doubles as the "never
//! sampled" marker — there is no separate sampled-pair set.

use std::collections::HashMap;

use crate::joint::config::JointSearchConfig;
use crate::joint::traits::{Evaluation, JointSnapshot};

/// Index of a node in its [`Tree`] arena; the root is always id 0.
pub type NodeId = u32;

/// One sampled chance outcome of a joint action pair.
#[derive(Debug)]
pub struct Outcome<S> {
    /// The successor position this outcome reached.
    pub snapshot: S,
    /// Potential difference vs the parent; exactly 0.0 for terminal
    /// outcomes, which are never shaped.
    pub tactical_delta: f64,
    /// The evaluator's unshaped value of the successor.
    pub leaf_value: f64,
}

/// A simultaneous-move tree node: a dense payoff matrix over joint
/// actions plus the warm RM+ solver state that lives between updates.
///
/// All action-indexed vectors are full `action_count` length (policies
/// start empty until the first solve); matrices are `action_count`²
/// row-major with rows indexed by the player action.
#[derive(Debug)]
pub struct TreeNode<S> {
    /// The position this node represents.
    pub snapshot: S,
    /// The evaluator's raw player priors, full `action_count` length.
    pub player_priors: Vec<f64>,
    /// The evaluator's raw enemy priors, same length.
    pub enemy_priors: Vec<f64>,
    /// The evaluator's unshaped value of this position.
    pub leaf_value: f64,
    /// Legal player actions in prior-descending order (index-ascending on
    /// ties), capped at `max_actions_per_side`. The order is load-bearing:
    /// argmax tie-breaks and rotation pairs both follow it.
    pub player_legal: Vec<usize>,
    /// Legal enemy actions, same ordering rules.
    pub enemy_legal: Vec<usize>,
    /// Running-mean payoff per joint-action cell.
    pub payoff: Vec<f64>,
    /// Values folded into each payoff cell so far.
    pub counts: Vec<u32>,
    /// Sampled outcomes per joint-action cell; an empty cell means the
    /// pair was never sampled.
    pub outcomes: Vec<Vec<Outcome<S>>>,
    /// Successor-snapshot id → child node, shared across chance outcomes
    /// that reach the same state.
    pub children: HashMap<u64, NodeId>,
    /// The player policy the last solve installed, scattered to full
    /// length; empty until the first
    /// [`solve_node`](crate::joint::solve_node).
    pub player_policy: Vec<f64>,
    /// The enemy policy the last solve installed, same contract.
    pub enemy_policy: Vec<f64>,
    /// Game value of the last solve, in the player's view.
    pub root_value: f64,
    /// Exploitability of the last solve's installed policies.
    pub exploitability: f64,
    /// Exploitability recorded at the node's first warm solve.
    pub online_exploitability: f64,
    /// Whether the node's payoff matrix has had outcomes installed.
    pub expanded: bool,
    /// Times the descent has passed through this node.
    pub visits: u32,
    /// Total warm RM+ iterations applied to this node so far.
    pub solve_count: u32,
    /// Player strategy accumulator across warm solves.
    pub player_strategy_sum: Vec<f64>,
    /// Enemy strategy accumulator across warm solves.
    pub enemy_strategy_sum: Vec<f64>,
    /// Persistent player regrets, warm-starting each solve.
    pub player_regrets: Vec<f64>,
    /// Persistent enemy regrets, same contract.
    pub enemy_regrets: Vec<f64>,
}

impl<S> TreeNode<S> {
    /// The full action-space size (the priors' length).
    pub fn action_count(&self) -> usize {
        self.player_priors.len()
    }

    fn cell(&self, player_action: usize, enemy_action: usize) -> usize {
        player_action * self.action_count() + enemy_action
    }

    /// The running-mean payoff of one joint-action cell.
    pub fn payoff_at(&self, player_action: usize, enemy_action: usize) -> f64 {
        self.payoff[self.cell(player_action, enemy_action)]
    }

    /// How many values the cell's running mean has folded in.
    pub fn count_at(&self, player_action: usize, enemy_action: usize) -> u32 {
        self.counts[self.cell(player_action, enemy_action)]
    }

    /// The sampled outcomes of one joint-action cell.
    pub fn outcomes_at(&self, player_action: usize, enemy_action: usize) -> &[Outcome<S>] {
        &self.outcomes[self.cell(player_action, enemy_action)]
    }

    /// Appends a sampled outcome to its joint-action cell.
    pub fn push_outcome(&mut self, player_action: usize, enemy_action: usize, outcome: Outcome<S>) {
        let cell = self.cell(player_action, enemy_action);
        self.outcomes[cell].push(outcome);
    }

    /// Folds `value` into the cell's running mean:
    /// `payoff = (payoff·count + value) / (count + 1)`, exactly this
    /// expression so repeat runs round identically.
    pub fn record_value(&mut self, player_action: usize, enemy_action: usize, value: f64) {
        let cell = self.cell(player_action, enemy_action);
        let count = self.counts[cell];
        let current = self.payoff[cell];
        self.payoff[cell] = (current * f64::from(count) + value) / f64::from(count + 1);
        self.counts[cell] = count + 1;
    }
}

/// Selects and orders one side's actions: mask bits within
/// `0..priors.len()` sorted by descending prior (ascending index on
/// ties), truncated to `max_actions_per_side`. Mask bits at or above the
/// action count are ignored.
pub fn legal_from_priors(mask: u64, priors: &[f64], max_actions_per_side: usize) -> Vec<usize> {
    assert!(priors.len() <= 64, "at most 64 actions are supported");
    let mut actions: Vec<usize> = (0..priors.len())
        .filter(|&action| mask & (1 << action) != 0)
        .collect();
    actions.sort_by(|&a, &b| {
        priors[b]
            .partial_cmp(&priors[a])
            .expect("action priors must be comparable")
            .then(a.cmp(&b))
    });
    actions.truncate(max_actions_per_side);
    actions
}

/// Truncates a prior-descending action list (as produced by
/// [`legal_from_priors`]) to the smallest prefix holding at least
/// `cutoff` of the list's total raw prior mass, never below `floor`
/// actions (clamped to the list length). A non-positive total counts
/// every action as equal mass, mirroring the uniform fallback of
/// [`normalized_prior`](crate::joint::normalized_prior).
///
/// Cumulative and total mass are summed in the same order, so a cutoff
/// of 1.0 keeps every positive-prior action exactly — but drops
/// zero-prior actions, the one difference from disabling the cutoff.
pub fn truncate_to_prior_mass(actions: &mut Vec<usize>, priors: &[f64], cutoff: f64, floor: usize) {
    let len = actions.len();
    if len == 0 {
        return;
    }
    let total: f64 = actions.iter().map(|&action| priors[action]).sum();
    let mut keep = len;
    if total > 0.0 {
        let target = cutoff * total;
        let mut cumulative = 0.0;
        for (index, &action) in actions.iter().enumerate() {
            cumulative += priors[action];
            if cumulative >= target {
                keep = index + 1;
                break;
            }
        }
    } else {
        let target = cutoff * len as f64;
        keep = (1..=len)
            .find(|&count| count as f64 >= target)
            .unwrap_or(len);
    }
    actions.truncate(keep.max(floor.min(len)));
}

/// One side's final legal list under `config`: prior-ordered, capped,
/// and mass-truncated when the pruning extension is enabled.
fn restricted_legal(mask: u64, priors: &[f64], config: &JointSearchConfig) -> Vec<usize> {
    let mut actions = legal_from_priors(mask, priors, config.max_actions_per_side);
    if let Some(cutoff) = config.prior_mass_cutoff {
        truncate_to_prior_mass(
            &mut actions,
            priors,
            cutoff,
            config.minimum_actions_per_side,
        );
    }
    actions
}

/// A search tree: nodes in creation order, the root first.
#[derive(Debug)]
pub struct Tree<S> {
    /// The nodes, indexed by [`NodeId`]; the root is `nodes[0]`.
    pub nodes: Vec<TreeNode<S>>,
    /// The full action-space size both sides share.
    pub action_count: usize,
}

impl<S: JointSnapshot> Tree<S> {
    /// Creates an empty tree for games with `action_count` actions per
    /// side. Panics outside `1..=64` — legality is a `u64` bitmask.
    pub fn new(action_count: usize) -> Self {
        assert!(
            (1..=64).contains(&action_count),
            "action count must be within 1..=64"
        );
        Self {
            nodes: Vec::new(),
            action_count,
        }
    }

    /// Builds a node from a snapshot and its evaluation and pushes it
    /// into the arena. The payoff matrix is prefilled with the
    /// leaf value on legal×legal cells only — unreachable cells stay 0 and
    /// never enter a solve. Panics when a side has no legal action: only
    /// non-terminal snapshots become nodes, and those must offer a move.
    pub fn make_node(
        &mut self,
        snapshot: S,
        evaluation: Evaluation,
        config: &JointSearchConfig,
    ) -> NodeId {
        let n = self.action_count;
        assert_eq!(evaluation.player_priors.len(), n, "player prior length");
        assert_eq!(evaluation.enemy_priors.len(), n, "enemy prior length");
        let player_legal =
            restricted_legal(snapshot.player_mask(), &evaluation.player_priors, config);
        let enemy_legal = restricted_legal(snapshot.enemy_mask(), &evaluation.enemy_priors, config);
        assert!(
            !player_legal.is_empty() && !enemy_legal.is_empty(),
            "a tree node needs at least one legal action per side"
        );
        let leaf_value = evaluation.value;
        debug_assert!(leaf_value.is_finite(), "leaf value must be finite");
        let mut payoff = vec![0.0; n * n];
        for &player_action in &player_legal {
            for &enemy_action in &enemy_legal {
                payoff[player_action * n + enemy_action] = leaf_value;
            }
        }
        let node = TreeNode {
            snapshot,
            player_priors: evaluation.player_priors,
            enemy_priors: evaluation.enemy_priors,
            leaf_value,
            player_legal,
            enemy_legal,
            payoff,
            counts: vec![0; n * n],
            outcomes: std::iter::repeat_with(Vec::new).take(n * n).collect(),
            children: HashMap::new(),
            player_policy: Vec::new(),
            enemy_policy: Vec::new(),
            root_value: 0.0,
            exploitability: 0.0,
            online_exploitability: 0.0,
            expanded: false,
            visits: 0,
            solve_count: 0,
            player_strategy_sum: vec![0.0; n],
            enemy_strategy_sum: vec![0.0; n],
            player_regrets: vec![0.0; n],
            enemy_regrets: vec![0.0; n],
        };
        let id = NodeId::try_from(self.nodes.len()).expect("tree node arena exceeds u32 ids");
        self.nodes.push(node);
        id
    }

    /// The root node. Panics while the tree is still empty.
    pub fn root(&self) -> &TreeNode<S> {
        &self.nodes[0]
    }

    /// The node with `id`. Panics when `id` is out of range.
    pub fn node(&self, id: NodeId) -> &TreeNode<S> {
        &self.nodes[id as usize]
    }

    /// Mutable access to the node with `id`.
    pub fn node_mut(&mut self, id: NodeId) -> &mut TreeNode<S> {
        &mut self.nodes[id as usize]
    }
}
