//! Toy providers and evaluators for the joint-search suites.
//!
//! Everything here is deliberately tiny and deterministic: matrix games
//! whose equilibria are known in closed form, wrappers that script
//! divergence or record the exact step requests, and evaluators with
//! fixed priors so legal-action ordering is under test control.

use std::collections::VecDeque;

use mcts_rs::joint::{
    legal_from_priors, strategy_weight_total, truncate_to_prior_mass, Divergence, Evaluation,
    Evaluator, JointSearchConfig, JointSnapshot, SearchResult, SolverTag, TransitionProvider, Tree,
};

/// A snapshot with every trait answer stored as plain data.
#[derive(Debug, Clone, PartialEq)]
pub struct ToySnapshot {
    pub id: u64,
    pub player_mask: u64,
    pub enemy_mask: u64,
    pub terminal: Option<f64>,
    pub potential: f64,
}

impl ToySnapshot {
    pub fn live(id: u64, player_mask: u64, enemy_mask: u64) -> Self {
        Self {
            id,
            player_mask,
            enemy_mask,
            terminal: None,
            potential: 0.0,
        }
    }

    pub fn terminal(id: u64, value: f64) -> Self {
        Self {
            id,
            player_mask: 0,
            enemy_mask: 0,
            terminal: Some(value),
            potential: 0.0,
        }
    }
}

impl JointSnapshot for ToySnapshot {
    fn id(&self) -> u64 {
        self.id
    }

    fn player_mask(&self) -> u64 {
        self.player_mask
    }

    fn enemy_mask(&self) -> u64 {
        self.enemy_mask
    }

    fn terminal_value(&self) -> Option<f64> {
        self.terminal
    }

    fn potential(&self) -> f64 {
        self.potential
    }
}

/// A one-shot matrix game: every joint action from the root ends the game
/// with the matrix payoff, regardless of the chance seed.
#[derive(Debug)]
pub struct MatrixProvider {
    pub action_count: usize,
    pub matrix: Vec<f64>,
}

impl MatrixProvider {
    pub fn new(action_count: usize, matrix: Vec<f64>) -> Self {
        assert_eq!(
            matrix.len(),
            action_count * action_count,
            "matrix must be action_count x action_count"
        );
        Self {
            action_count,
            matrix,
        }
    }

    /// The root snapshot: all actions legal on both sides.
    pub fn root(&self) -> ToySnapshot {
        let mask = (1u64 << self.action_count) - 1;
        ToySnapshot::live(0, mask, mask)
    }
}

impl TransitionProvider for MatrixProvider {
    type Snapshot = ToySnapshot;

    fn step(
        &mut self,
        _parent: &ToySnapshot,
        player_action: usize,
        enemy_action: usize,
        _chance_seed: u64,
    ) -> Result<ToySnapshot, Divergence> {
        let cell = player_action * self.action_count + enemy_action;
        Ok(ToySnapshot::terminal(1 + cell as u64, self.matrix[cell]))
    }
}

/// Delegates to an inner provider, failing every step past a scripted
/// number of successes.
#[derive(Debug)]
pub struct DivergeAfter<P> {
    pub inner: P,
    pub successes: u32,
    pub steps: u32,
}

impl<P> DivergeAfter<P> {
    pub fn new(inner: P, successes: u32) -> Self {
        Self {
            inner,
            successes,
            steps: 0,
        }
    }
}

impl<P: TransitionProvider> TransitionProvider for DivergeAfter<P> {
    type Snapshot = P::Snapshot;

    fn step(
        &mut self,
        parent: &P::Snapshot,
        player_action: usize,
        enemy_action: usize,
        chance_seed: u64,
    ) -> Result<P::Snapshot, Divergence> {
        if self.steps >= self.successes {
            return Err(Divergence::new("scripted divergence"));
        }
        self.steps += 1;
        self.inner
            .step(parent, player_action, enemy_action, chance_seed)
    }
}

/// Delegates to an inner provider, logging every request as
/// `(parent id, player action, enemy action, chance seed)`.
#[derive(Debug)]
pub struct RecordingProvider<P: TransitionProvider> {
    pub inner: P,
    pub log: Vec<(u64, usize, usize, u64)>,
}

impl<P: TransitionProvider> RecordingProvider<P> {
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            log: Vec::new(),
        }
    }
}

impl<P: TransitionProvider> TransitionProvider for RecordingProvider<P> {
    type Snapshot = P::Snapshot;

    fn step(
        &mut self,
        parent: &P::Snapshot,
        player_action: usize,
        enemy_action: usize,
        chance_seed: u64,
    ) -> Result<P::Snapshot, Divergence> {
        self.log
            .push((parent.id(), player_action, enemy_action, chance_seed));
        self.inner
            .step(parent, player_action, enemy_action, chance_seed)
    }
}

/// A two-ply game with two actions per side: every root joint action
/// leads to the same second-stage matrix game (snapshot id 1), except
/// that player action 1 at the root can be scripted to end the game
/// immediately at `bail_value`. The stage snapshot carries
/// `stage_potential`, so potential shaping is exercised end to end, and
/// every root pair reaching the same stage id exercises child sharing
/// across chance outcomes.
#[derive(Debug)]
pub struct TwoStage {
    /// Row-major 2x2 terminal payoffs of the second stage.
    pub stage_matrix: Vec<f64>,
    pub stage_potential: f64,
    /// `Some(v)`: player action 1 at the root ends the game at `v`.
    pub bail_value: Option<f64>,
}

impl TwoStage {
    pub fn root() -> ToySnapshot {
        ToySnapshot::live(0, 0b11, 0b11)
    }
}

impl TransitionProvider for TwoStage {
    type Snapshot = ToySnapshot;

    fn step(
        &mut self,
        parent: &ToySnapshot,
        player_action: usize,
        enemy_action: usize,
        _chance_seed: u64,
    ) -> Result<ToySnapshot, Divergence> {
        match parent.id {
            0 => {
                if let (Some(value), 1) = (self.bail_value, player_action) {
                    Ok(ToySnapshot::terminal(2, value))
                } else {
                    Ok(ToySnapshot {
                        id: 1,
                        player_mask: 0b11,
                        enemy_mask: 0b11,
                        terminal: None,
                        potential: self.stage_potential,
                    })
                }
            }
            1 => {
                let cell = player_action * 2 + enemy_action;
                Ok(ToySnapshot::terminal(
                    3 + cell as u64,
                    self.stage_matrix[cell],
                ))
            }
            id => panic!("TwoStage stepped an unexpected snapshot {id}"),
        }
    }
}

/// Every joint action ends the game immediately, with the payoff sign
/// decided by the chance seed's parity — the same pair yields different
/// outcomes across resamples, so evidence genuinely accumulates.
#[derive(Debug)]
pub struct SeedSensitiveProvider;

impl SeedSensitiveProvider {
    pub fn root() -> ToySnapshot {
        ToySnapshot::live(0, 0b11, 0b11)
    }
}

impl TransitionProvider for SeedSensitiveProvider {
    type Snapshot = ToySnapshot;

    fn step(
        &mut self,
        _parent: &ToySnapshot,
        _player_action: usize,
        _enemy_action: usize,
        chance_seed: u64,
    ) -> Result<ToySnapshot, Divergence> {
        if chance_seed & 1 == 0 {
            Ok(ToySnapshot::terminal(1, 1.0))
        } else {
            Ok(ToySnapshot::terminal(2, -1.0))
        }
    }
}

/// Uniform priors over every action and one fixed value everywhere.
#[derive(Debug)]
pub struct UniformEvaluator {
    pub action_count: usize,
    pub value: f64,
}

impl Evaluator<ToySnapshot> for UniformEvaluator {
    fn action_count(&self) -> usize {
        self.action_count
    }

    fn evaluate(&mut self, _snapshot: &ToySnapshot) -> Evaluation {
        let prior = 1.0 / self.action_count as f64;
        Evaluation {
            player_priors: vec![prior; self.action_count],
            enemy_priors: vec![prior; self.action_count],
            value: self.value,
        }
    }
}

/// Fixed per-side priors and one fixed value everywhere; the prior order
/// decides the legal-action order, so tests pick these deliberately.
#[derive(Debug)]
pub struct FixedPriorEvaluator {
    pub player_priors: Vec<f64>,
    pub enemy_priors: Vec<f64>,
    pub value: f64,
}

impl Evaluator<ToySnapshot> for FixedPriorEvaluator {
    fn action_count(&self) -> usize {
        self.player_priors.len()
    }

    fn evaluate(&mut self, _snapshot: &ToySnapshot) -> Evaluation {
        Evaluation {
            player_priors: self.player_priors.clone(),
            enemy_priors: self.enemy_priors.clone(),
            value: self.value,
        }
    }
}

/// Checks every structural invariant a successful `search_with_tree` run
/// must leave behind, for any provider, evaluator, and configuration:
///
/// - child linkage: every node is reachable from the root through exactly
///   one parent, children are keyed by their snapshot id, and terminal
///   snapshots never become nodes;
/// - legal lists reconstruct from the snapshot masks and priors (masked,
///   deduped, capped, prior-descending order);
/// - solver state: regrets non-negative and zero off-legal, strategy sums
///   zero off-legal with total mass tracking the solve count's
///   `strategy_weight_total` (the plain count, or the triangular number
///   under `cfr_plus_solves` — which also pins the linear weights
///   continuing globally across warm batches), solves in whole warm
///   batches, policies full-length distributions over legal actions,
///   unexpanded nodes fully zeroed, and — under
///   `average_strategy_policies` — every non-root expanded node's policy
///   is the cumulative `strategy_sum / strategy_weight_total` bitwise
///   (the root's policy is always the cold root equilibrium);
/// - matrix cells: counts bound the outcome lists, unsampled cells keep
///   the leaf prefill bitwise, off-grid cells are untouched, terminal
///   outcomes are never potential-shaped, and cells that resolve without
///   descending (terminal or depth-capped) replay the running mean
///   bitwise;
/// - counters: transitions == chance outcomes == outcomes stored
///   tree-wide, coverage recomputes from sampled/possible pairs, the
///   budget is never exceeded past the root install, and the reached
///   depth stays under the depth cap;
/// - result surface: the result mirrors the root node (policies, value,
///   exploitability, payoff matrix, spread) and both chosen actions are
///   legal.
///
/// `ctx` is prefixed to every failure message to identify the scenario.
pub fn assert_joint_tree_invariants<S: JointSnapshot>(
    tree: &Tree<S>,
    result: &SearchResult,
    config: &JointSearchConfig,
    ctx: &str,
) {
    assert!(
        result.failure.is_none(),
        "{ctx}: the checker needs a successful result"
    );
    assert_eq!(
        result.solver,
        SolverTag::RmPlusPooledNodeV3,
        "{ctx}: solver tag"
    );
    let n = tree.action_count;

    // Depths and child linkage via BFS over the children maps.
    let mut depths: Vec<Option<u32>> = vec![None; tree.nodes.len()];
    depths[0] = Some(0);
    let mut queue = VecDeque::from([0u32]);
    while let Some(id) = queue.pop_front() {
        let node = tree.node(id);
        let node_depth = depths[id as usize].expect("queued nodes have depths");
        for (&snapshot_id, &child_id) in &node.children {
            assert!(
                depths[child_id as usize].is_none(),
                "{ctx}: node {child_id} is linked by two parents"
            );
            depths[child_id as usize] = Some(node_depth + 1);
            let child = tree.node(child_id);
            assert_eq!(
                child.snapshot.id(),
                snapshot_id,
                "{ctx}: children are keyed by their snapshot id"
            );
            assert!(
                child.snapshot.terminal_value().is_none(),
                "{ctx}: terminal snapshots never become nodes"
            );
            queue.push_back(child_id);
        }
    }

    let mut total_outcomes = 0u32;
    let mut expanded_nodes = 0u32;
    let mut sampled_pairs = 0u32;
    let mut possible_pairs = 0u32;
    let mut deepest = 0u32;

    for (index, node) in tree.nodes.iter().enumerate() {
        let depth = depths[index].unwrap_or_else(|| panic!("{ctx}: node {index} is orphaned"));

        for (side, legal, mask, priors, regrets, sums, policy) in [
            (
                "player",
                &node.player_legal,
                node.snapshot.player_mask(),
                &node.player_priors,
                &node.player_regrets,
                &node.player_strategy_sum,
                &node.player_policy,
            ),
            (
                "enemy",
                &node.enemy_legal,
                node.snapshot.enemy_mask(),
                &node.enemy_priors,
                &node.enemy_regrets,
                &node.enemy_strategy_sum,
                &node.enemy_policy,
            ),
        ] {
            let mut expected_legal = legal_from_priors(mask, priors, config.max_actions_per_side);
            if let Some(cutoff) = config.prior_mass_cutoff {
                truncate_to_prior_mass(
                    &mut expected_legal,
                    priors,
                    cutoff,
                    config.minimum_actions_per_side,
                );
            }
            assert_eq!(
                *legal, expected_legal,
                "{ctx}: node {index} {side} legal actions reconstruct from the mask and priors"
            );
            assert_eq!(priors.len(), n, "{ctx}: node {index} {side} prior length");
            assert_eq!(regrets.len(), n, "{ctx}: node {index} {side} regret length");
            assert_eq!(
                sums.len(),
                n,
                "{ctx}: node {index} {side} strategy-sum length"
            );
            let mut strategy_mass = 0.0;
            for action in 0..n {
                assert!(
                    regrets[action] >= 0.0,
                    "{ctx}: node {index} {side} RM+ regrets stay non-negative"
                );
                if !legal.contains(&action) {
                    assert_eq!(
                        regrets[action], 0.0,
                        "{ctx}: node {index} {side} keeps no regret off-legal"
                    );
                    assert_eq!(
                        sums[action], 0.0,
                        "{ctx}: node {index} {side} keeps no strategy mass off-legal"
                    );
                }
                strategy_mass += sums[action];
            }
            let weight_total = strategy_weight_total(config.cfr_plus_solves, node.solve_count);
            assert!(
                (strategy_mass - weight_total).abs() <= 1e-9 * weight_total.max(1.0),
                "{ctx}: node {index} {side} strategy mass {strategy_mass} must track the \
                 accumulated strategy weight {weight_total}"
            );
            if node.expanded {
                assert_eq!(policy.len(), n, "{ctx}: node {index} {side} policy length");
                let mut policy_mass = 0.0;
                for (action, &mass) in policy.iter().enumerate() {
                    assert!(
                        mass >= 0.0,
                        "{ctx}: node {index} {side} policies are non-negative"
                    );
                    if !legal.contains(&action) {
                        assert_eq!(
                            mass, 0.0,
                            "{ctx}: node {index} {side} puts no policy mass off-legal"
                        );
                    }
                    policy_mass += mass;
                }
                assert!(
                    (policy_mass - 1.0).abs() <= 1e-9,
                    "{ctx}: node {index} {side} policy must sum to one, got {policy_mass}"
                );
                if config.average_strategy_policies && index != 0 {
                    for &action in legal.iter() {
                        assert_eq!(
                            policy[action].to_bits(),
                            (sums[action] / weight_total).to_bits(),
                            "{ctx}: node {index} {side} action {action} policy must be \
                             the cumulative average strategy"
                        );
                    }
                }
            } else {
                assert!(
                    policy.is_empty(),
                    "{ctx}: node {index} {side} must not have a policy before its first solve"
                );
            }
        }

        assert_eq!(
            node.solve_count % config.regret_iterations_per_update,
            0,
            "{ctx}: node {index} solves in whole warm batches"
        );
        if node.visits > 0 {
            assert!(
                node.expanded,
                "{ctx}: node {index} was simulated before expansion"
            );
        }
        if node.expanded {
            expanded_nodes += 1;
            possible_pairs += u32::try_from(node.player_legal.len() * node.enemy_legal.len())
                .expect("pair count fits u32");
            assert!(
                node.exploitability >= -1e-9,
                "{ctx}: node {index} exploitability must be non-negative"
            );
            deepest = deepest.max(depth);
            for &player in &node.player_legal {
                assert!(
                    node.enemy_legal
                        .iter()
                        .any(|&enemy| !node.outcomes_at(player, enemy).is_empty()),
                    "{ctx}: node {index} player action {player} appears in no sampled pair"
                );
            }
            for &enemy in &node.enemy_legal {
                assert!(
                    node.player_legal
                        .iter()
                        .any(|&player| !node.outcomes_at(player, enemy).is_empty()),
                    "{ctx}: node {index} enemy action {enemy} appears in no sampled pair"
                );
            }
        } else {
            assert_eq!(
                node.solve_count, 0,
                "{ctx}: node {index} must not be solved before expansion"
            );
            for value in node
                .player_regrets
                .iter()
                .chain(&node.enemy_regrets)
                .chain(&node.player_strategy_sum)
                .chain(&node.enemy_strategy_sum)
            {
                assert_eq!(
                    *value, 0.0,
                    "{ctx}: node {index} unexpanded solver state must be zeroed"
                );
            }
        }

        for player in 0..n {
            for enemy in 0..n {
                let count = node.count_at(player, enemy);
                let outcomes = node.outcomes_at(player, enemy);
                total_outcomes += u32::try_from(outcomes.len()).expect("outcome count fits u32");
                if !outcomes.is_empty() {
                    sampled_pairs += 1;
                }
                if !node.expanded {
                    assert_eq!(
                        count, 0,
                        "{ctx}: node {index} cell ({player}, {enemy}) recorded before expansion"
                    );
                }
                let legal_cell =
                    node.player_legal.contains(&player) && node.enemy_legal.contains(&enemy);
                if !legal_cell {
                    assert_eq!(
                        count, 0,
                        "{ctx}: node {index} off-grid cell ({player}, {enemy}) was recorded"
                    );
                    assert!(
                        outcomes.is_empty(),
                        "{ctx}: node {index} off-grid cell ({player}, {enemy}) was sampled"
                    );
                    assert_eq!(
                        node.payoff_at(player, enemy),
                        0.0,
                        "{ctx}: node {index} off-grid cell ({player}, {enemy}) payoff touched"
                    );
                }
                assert!(
                    count as usize >= outcomes.len(),
                    "{ctx}: node {index} cell ({player}, {enemy}) has more outcomes than records"
                );
                assert_eq!(
                    count > 0,
                    !outcomes.is_empty(),
                    "{ctx}: node {index} cell ({player}, {enemy}) records and outcomes go together"
                );
                if legal_cell && count == 0 {
                    assert_eq!(
                        node.payoff_at(player, enemy),
                        node.leaf_value,
                        "{ctx}: node {index} unsampled cell ({player}, {enemy}) keeps the prefill"
                    );
                }
                for outcome in outcomes {
                    if let Some(terminal) = outcome.snapshot.terminal_value() {
                        assert_eq!(
                            outcome.tactical_delta, 0.0,
                            "{ctx}: node {index} terminal outcome shaped at ({player}, {enemy})"
                        );
                        assert_eq!(
                            outcome.leaf_value, terminal,
                            "{ctx}: node {index} terminal return replaces the evaluator value"
                        );
                    }
                }
                // Cells that never descend through a child record exactly
                // the shaped outcome value once per sample, in sample
                // order, so the running mean replays bitwise.
                let descends = depth + 1 < config.max_depth
                    && outcomes
                        .iter()
                        .any(|outcome| outcome.snapshot.terminal_value().is_none());
                if count > 0 && !descends {
                    assert_eq!(
                        count as usize,
                        outcomes.len(),
                        "{ctx}: node {index} leaf cell ({player}, {enemy}) records once per outcome"
                    );
                    let mut mean = 0.0;
                    for (recorded, outcome) in outcomes.iter().enumerate() {
                        let shaped = if outcome.snapshot.terminal_value().is_some() {
                            outcome.leaf_value
                        } else {
                            (outcome.leaf_value + outcome.tactical_delta).clamp(-1.0, 1.0)
                        };
                        let recorded = u32::try_from(recorded).expect("record count fits u32");
                        mean = (mean * f64::from(recorded) + shaped) / f64::from(recorded + 1);
                    }
                    assert_eq!(
                        node.payoff_at(player, enemy),
                        mean,
                        "{ctx}: node {index} cell ({player}, {enemy}) running mean must replay"
                    );
                }
            }
        }
    }

    let diagnostics = &result.diagnostics;
    assert_eq!(
        diagnostics.tree_nodes, expanded_nodes,
        "{ctx}: tree_nodes counts expanded nodes"
    );
    assert_eq!(
        diagnostics.chance_outcomes, total_outcomes,
        "{ctx}: chance_outcomes counts stored outcomes"
    );
    assert_eq!(
        result.transitions, total_outcomes,
        "{ctx}: every transition is stored as an outcome"
    );
    assert_eq!(
        diagnostics.tree_max_depth, deepest,
        "{ctx}: tree_max_depth is the deepest expanded node"
    );
    assert!(
        diagnostics.tree_max_depth < config.max_depth,
        "{ctx}: descent respects the depth cap"
    );
    let coverage = if possible_pairs > 0 {
        f64::from(sampled_pairs) / f64::from(possible_pairs)
    } else {
        0.0
    };
    assert_eq!(
        diagnostics.sampled_joint_coverage, coverage,
        "{ctx}: joint coverage recomputes from the tree"
    );

    let root = tree.root();
    let root_install = u32::try_from(root.player_legal.len() * root.enemy_legal.len())
        .expect("pair count fits u32")
        * config.chance_samples_per_joint;
    assert!(
        result.transitions <= config.expansion_budget.max(root_install),
        "{ctx}: the budget is never exceeded past the root install"
    );

    let root_diagnostics = diagnostics
        .root
        .as_ref()
        .unwrap_or_else(|| panic!("{ctx}: successful searches report root diagnostics"));
    assert_eq!(
        root_diagnostics.joint_actions,
        u32::try_from(root.player_legal.len() * root.enemy_legal.len()).expect("pair count fits u32"),
        "{ctx}: root joint actions"
    );
    assert_eq!(
        root_diagnostics.solves, root.solve_count,
        "{ctx}: root solve count"
    );
    assert_eq!(
        root_diagnostics.online_exploitability, root.online_exploitability,
        "{ctx}: root online exploitability"
    );
    assert_eq!(
        root_diagnostics.final_exploitability, root.exploitability,
        "{ctx}: root final exploitability"
    );
    assert_eq!(
        root_diagnostics.equilibrium_iterations, config.regret_iterations,
        "{ctx}: root equilibrium iterations"
    );

    assert_eq!(
        result.player_policy, root.player_policy,
        "{ctx}: the result mirrors the root player policy"
    );
    assert_eq!(
        result.enemy_policy, root.enemy_policy,
        "{ctx}: the result mirrors the root enemy policy"
    );
    assert_eq!(
        result.root_value, root.root_value,
        "{ctx}: the result mirrors the root value"
    );
    assert_eq!(
        result.exploitability,
        Some(root.exploitability),
        "{ctx}: the result mirrors the root exploitability"
    );
    assert_eq!(
        result.payoff_matrix.as_deref(),
        Some(root.payoff.as_slice()),
        "{ctx}: the result carries the root payoff matrix"
    );
    let mut max_payoff = f64::NEG_INFINITY;
    let mut min_payoff = f64::INFINITY;
    for &player in &root.player_legal {
        for &enemy in &root.enemy_legal {
            max_payoff = max_payoff.max(root.payoff_at(player, enemy));
            min_payoff = min_payoff.min(root.payoff_at(player, enemy));
        }
    }
    assert_eq!(
        result.payoff_spread,
        Some(max_payoff - min_payoff),
        "{ctx}: payoff spread over legal root cells"
    );
    let player_action = result
        .player_action
        .unwrap_or_else(|| panic!("{ctx}: successful searches choose a player action"));
    let enemy_action = result
        .enemy_action
        .unwrap_or_else(|| panic!("{ctx}: successful searches choose an enemy action"));
    assert!(
        root.player_legal.contains(&player_action),
        "{ctx}: the chosen player action is legal"
    );
    assert!(
        root.enemy_legal.contains(&enemy_action),
        "{ctx}: the chosen enemy action is legal"
    );
}
