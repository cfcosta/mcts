//! The joint search tree: matrix-bearing nodes in a flat arena.
//!
//! Ports `_TreeNode`/`_TreeOutcome` and the node-construction helpers from
//! the Python search. Nodes live in a plain `Vec` indexed by [`NodeId`]
//! (root = 0) rather than a bump arena: a joint tree holds tens of heavy
//! matrix-bearing nodes, not the tens of thousands of tiny nodes the UCT
//! arena is built for. The per-cell state Python keeps as nested lists and
//! dicts is flattened to row-major `action_count`² vectors; a cell's
//! outcome list being empty stands in for Python's separate
//! `sampled_pairs` set, which always mirrored the outcome keys.

use std::collections::HashMap;

use crate::joint::config::JointSearchConfig;
use crate::joint::traits::{Evaluation, JointSnapshot};

/// Index of a node in its [`Tree`] arena; the root is always id 0.
pub type NodeId = u32;

/// One sampled chance outcome of a joint action pair (`_TreeOutcome`).
#[derive(Debug)]
pub struct Outcome<S> {
    pub snapshot: S,
    /// Potential difference vs the parent; exactly 0.0 for terminal
    /// outcomes, which are never shaped.
    pub tactical_delta: f64,
    /// The evaluator's unshaped value of the successor.
    pub leaf_value: f64,
}

/// A simultaneous-move tree node (`_TreeNode`): a dense payoff matrix over
/// joint actions plus the warm RM+ solver state that lives between
/// updates.
///
/// All action-indexed vectors are full `action_count` length (policies
/// start empty, exactly like Python's `()` default, until the first
/// solve); matrices are `action_count`² row-major with rows indexed by the
/// player action.
#[derive(Debug)]
pub struct TreeNode<S> {
    pub snapshot: S,
    pub player_priors: Vec<f64>,
    pub enemy_priors: Vec<f64>,
    pub leaf_value: f64,
    /// Legal actions in prior-descending order (index-ascending on ties),
    /// capped at `max_actions_per_side`. The order is load-bearing: argmax
    /// tie-breaks and rotation pairs both follow it.
    pub player_legal: Vec<usize>,
    pub enemy_legal: Vec<usize>,
    pub payoff: Vec<f64>,
    pub counts: Vec<u32>,
    /// Sampled outcomes per joint-action cell; an empty cell means the
    /// pair was never sampled.
    pub outcomes: Vec<Vec<Outcome<S>>>,
    /// Successor-snapshot id → child node, shared across chance outcomes
    /// that reach the same state.
    pub children: HashMap<u64, NodeId>,
    /// Last-iterate solve output, scattered to full length; empty until
    /// the first [`solve_node`](crate::joint::solve_node).
    pub player_policy: Vec<f64>,
    pub enemy_policy: Vec<f64>,
    pub root_value: f64,
    pub exploitability: f64,
    pub online_exploitability: f64,
    pub expanded: bool,
    pub visits: u32,
    pub solve_count: u32,
    pub player_strategy_sum: Vec<f64>,
    pub enemy_strategy_sum: Vec<f64>,
    pub player_regrets: Vec<f64>,
    pub enemy_regrets: Vec<f64>,
}

impl<S> TreeNode<S> {
    pub fn action_count(&self) -> usize {
        self.player_priors.len()
    }

    fn cell(&self, player_action: usize, enemy_action: usize) -> usize {
        player_action * self.action_count() + enemy_action
    }

    pub fn payoff_at(&self, player_action: usize, enemy_action: usize) -> f64 {
        self.payoff[self.cell(player_action, enemy_action)]
    }

    pub fn count_at(&self, player_action: usize, enemy_action: usize) -> u32 {
        self.counts[self.cell(player_action, enemy_action)]
    }

    pub fn outcomes_at(&self, player_action: usize, enemy_action: usize) -> &[Outcome<S>] {
        &self.outcomes[self.cell(player_action, enemy_action)]
    }

    pub fn push_outcome(&mut self, player_action: usize, enemy_action: usize, outcome: Outcome<S>) {
        let cell = self.cell(player_action, enemy_action);
        self.outcomes[cell].push(outcome);
    }

    /// Folds `value` into the cell's running mean (`_record_value`):
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

/// Selects and orders one side's actions (`_legal_from_priors`): mask bits
/// within `0..priors.len()` sorted by descending prior (ascending index on
/// ties), truncated to `max_actions_per_side`. Bits at or above the action
/// count are ignored, exactly like Python's `range(action_count)` scan.
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

/// A search tree: nodes in creation order, the root first.
#[derive(Debug)]
pub struct Tree<S> {
    pub nodes: Vec<TreeNode<S>>,
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

    /// Builds a node from a snapshot and its evaluation (`_make_node`) and
    /// pushes it into the arena. The payoff matrix is prefilled with the
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
        let player_legal = legal_from_priors(
            snapshot.player_mask(),
            &evaluation.player_priors,
            config.max_actions_per_side,
        );
        let enemy_legal = legal_from_priors(
            snapshot.enemy_mask(),
            &evaluation.enemy_priors,
            config.max_actions_per_side,
        );
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

    pub fn root(&self) -> &TreeNode<S> {
        &self.nodes[0]
    }

    pub fn node(&self, id: NodeId) -> &TreeNode<S> {
        &self.nodes[id as usize]
    }

    pub fn node_mut(&mut self, id: NodeId) -> &mut TreeNode<S> {
        &mut self.nodes[id as usize]
    }
}
