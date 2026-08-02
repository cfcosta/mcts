//! The search driver: root installation and the equilibrium pipeline.
//!
//! This is the engine half of the port — the single-root path of the
//! Python `search`/`search_many` plus `_expansion_spec`,
//! `_install_expansion`, `_choose_adaptive_depth` and the assembly part
//! of `_finish_root_frontier`: evaluate the root, install the full joint
//! grid under common random numbers, warm-solve it, overwrite the node
//! with the cold equilibrium, route deep or shallow, and assemble the
//! result. The deep descent loop between routing and assembly arrives
//! with the descent milestone; until then a deep-routed search proceeds
//! straight to final assembly, which matches the full semantics whenever
//! the root install already consumes the expansion budget.

use rand::RngCore;

use crate::joint::config::JointSearchConfig;
use crate::joint::node::{NodeId, Outcome, Tree, TreeNode};
use crate::joint::result::{
    AdaptiveReason, Diagnostics, RootDiagnostics, SearchOptions, SearchResult, SolverTag,
};
use crate::joint::rng::{next_f64, SplitMix64};
use crate::joint::solver::{
    argmax_first, expansion_pairs, policy_entropy, sample_index, solve_node, solve_zero_sum_regret,
};
use crate::joint::traits::{Divergence, Evaluator, JointSnapshot, TransitionProvider};

/// The simultaneous-move regret-matching tree search.
///
/// Holds the configuration and the three independent random streams the
/// Python search keeps as separate `random.Random` instances: `selection`
/// (descent sampling and final action draws), `chance` (transition
/// seeds), and `budget` (the forced-calibration coin of the adaptive
/// router). Which stream serves each draw, and the order of draws within
/// a stream, are part of the ported semantics; the streams themselves are
/// [`SplitMix64`] rather than CPython's Mersenne Twister, so seeds do not
/// reproduce Python runs — only Rust runs.
///
/// The search object carries no per-call state: every [`search`] call is
/// the Python pattern of a fresh search object reusing persistent RNGs.
///
/// [`search`]: SimultaneousTreeSearch::search
#[derive(Debug)]
pub struct SimultaneousTreeSearch<R: RngCore = SplitMix64> {
    config: JointSearchConfig,
    selection_rng: R,
    chance_rng: R,
    budget_rng: R,
}

/// Per-search bookkeeping: the Python search object's counters and
/// adaptive verdicts, reset for every root exactly as the pipeline
/// constructs a fresh search per episode turn.
struct SearchRun {
    expanded_nodes: u32,
    chance_outcomes: u32,
    max_depth_reached: u32,
    sampled_joint_pairs: u32,
    possible_joint_pairs: u32,
    adaptive_deep_selected: bool,
    adaptive_router_score: f64,
    adaptive_reason: AdaptiveReason,
    deep_policy_change: f64,
    deep_action_changed: bool,
    deep_search_needed: Option<bool>,
}

impl SearchRun {
    fn new() -> Self {
        Self {
            expanded_nodes: 0,
            chance_outcomes: 0,
            max_depth_reached: 0,
            sampled_joint_pairs: 0,
            possible_joint_pairs: 0,
            adaptive_deep_selected: true,
            adaptive_router_score: 1.0,
            adaptive_reason: AdaptiveReason::Disabled,
            deep_policy_change: 0.0,
            deep_action_changed: false,
            deep_search_needed: None,
        }
    }
}

impl SimultaneousTreeSearch<SplitMix64> {
    /// Creates a search with three [`SplitMix64`] streams derived from one
    /// seed. Panics when the configuration is invalid.
    pub fn new(config: JointSearchConfig, seed: u64) -> Self {
        let mut seeder = SplitMix64::new(seed);
        let selection_rng = SplitMix64::new(seeder.next_u64());
        let chance_rng = SplitMix64::new(seeder.next_u64());
        let budget_rng = SplitMix64::new(seeder.next_u64());
        Self::with_rngs(config, selection_rng, chance_rng, budget_rng)
    }
}

impl<R: RngCore> SimultaneousTreeSearch<R> {
    /// Creates a search from explicit random streams, mirroring how the
    /// pipeline threads per-episode RNGs into each fresh search object.
    /// Panics when the configuration is invalid.
    pub fn with_rngs(
        config: JointSearchConfig,
        selection_rng: R,
        chance_rng: R,
        budget_rng: R,
    ) -> Self {
        if let Err(error) = config.validate() {
            panic!("invalid search config: {error}");
        }
        Self {
            config,
            selection_rng,
            chance_rng,
            budget_rng,
        }
    }

    pub fn config(&self) -> &JointSearchConfig {
        &self.config
    }

    /// Runs a search from `root` and returns the result.
    ///
    /// Panics when `root` is terminal — searching a finished position is
    /// a caller error, exactly as in Python.
    pub fn search<P, E>(
        &mut self,
        provider: &mut P,
        evaluator: &mut E,
        root: P::Snapshot,
        options: SearchOptions,
    ) -> SearchResult
    where
        P: TransitionProvider,
        E: Evaluator<P::Snapshot>,
    {
        self.search_with_tree(provider, evaluator, root, options).0
    }

    /// [`search`](SimultaneousTreeSearch::search), also returning the tree
    /// so tests can inspect every node the search left behind.
    pub fn search_with_tree<P, E>(
        &mut self,
        provider: &mut P,
        evaluator: &mut E,
        root: P::Snapshot,
        options: SearchOptions,
    ) -> (SearchResult, Tree<P::Snapshot>)
    where
        P: TransitionProvider,
        E: Evaluator<P::Snapshot>,
    {
        let action_count = evaluator.action_count();
        assert!(
            root.terminal_value().is_none(),
            "cannot search a terminal state"
        );
        let mut tree = Tree::new(action_count);
        let mut run = SearchRun::new();
        let evaluation = evaluator.evaluate(&root);
        let prior_value = evaluation.value;
        let root_id = tree.make_node(root, evaluation, &self.config);

        // Root install: the full joint grid, exempt from the expansion
        // budget (the budget gates only the descent loop).
        let transitions = match self.install_root(&mut run, &mut tree, root_id, provider, evaluator) {
            Ok(transitions) => transitions,
            Err((divergence, steps)) => {
                let result = self.fallback(&run, action_count, prior_value, steps, 0, divergence);
                return (result, tree);
            }
        };
        {
            let node = tree.node_mut(root_id);
            node.online_exploitability = node.exploitability;
            let (player_policy, enemy_policy, value, exploitability) =
                root_equilibrium(node, self.config.regret_iterations);
            node.player_policy = player_policy;
            node.enemy_policy = enemy_policy;
            node.root_value = value;
            node.exploitability = exploitability;
        }
        // Python records 1.0 for non-adaptive searches regardless of the
        // caller's score (the pooled router only scores adaptive roots).
        let router_score = if self.config.adaptive_search {
            options.router_score
        } else {
            1.0
        };
        let reason = self.choose_adaptive_depth(tree.node(root_id), router_score);
        run.adaptive_deep_selected = reason.is_deep();
        run.adaptive_router_score = router_score;
        run.adaptive_reason = reason;
        let result = self.finish_root(&mut run, &mut tree, root_id, transitions, options);
        (result, tree)
    }

    /// Expands the root over its full legal grid (`_expansion_spec` with
    /// `full_matrix=True` plus the stepping the pipeline pools): one
    /// chance seed per sample index shared across every pair — common
    /// random numbers — drawn from the chance stream. On divergence,
    /// returns the failing step's error and the number of step calls made
    /// (Python reports the size of the whole pooled batch instead; this
    /// deviation is deliberate and documented).
    fn install_root<P, E>(
        &mut self,
        run: &mut SearchRun,
        tree: &mut Tree<P::Snapshot>,
        node_id: NodeId,
        provider: &mut P,
        evaluator: &mut E,
    ) -> Result<u32, (Divergence, u32)>
    where
        P: TransitionProvider,
        E: Evaluator<P::Snapshot>,
    {
        let samples: Vec<(usize, usize, u64)> = {
            let node = tree.node(node_id);
            let pairs = expansion_pairs(
                &node.player_legal,
                &node.enemy_legal,
                true,
                self.config.deeper_joint_rotations,
            );
            let seeds: Vec<u64> = (0..self.config.chance_samples_per_joint)
                .map(|_| self.chance_rng.next_u64())
                .collect();
            // Pair-major, sample-minor: the Python request order.
            pairs
                .iter()
                .flat_map(|&(player, enemy)| seeds.iter().map(move |&seed| (player, enemy, seed)))
                .collect()
        };
        let mut successors = Vec::with_capacity(samples.len());
        let mut steps = 0u32;
        for &(player, enemy, seed) in &samples {
            steps += 1;
            match provider.step(&tree.node(node_id).snapshot, player, enemy, seed) {
                Ok(successor) => successors.push(successor),
                Err(divergence) => return Err((divergence, steps)),
            }
        }
        // Terminal successors keep their raw return; only live successors
        // consult the evaluator (Python substitutes `terminal_return`
        // centrally in `_pooled_successor_values`).
        let leaf_values: Vec<f64> = successors
            .iter()
            .map(|successor| match successor.terminal_value() {
                Some(value) => value,
                None => evaluator.leaf_value(successor),
            })
            .collect();
        Ok(self.install_expansion(run, tree, node_id, &samples, successors, &leaf_values))
    }

    /// Installs stepped-and-evaluated outcomes into a node
    /// (`_install_expansion`): shape each leaf value by the potential
    /// difference (terminal outcomes are never shaped **or clamped**),
    /// record it into the running-mean payoff, then warm-solve the node.
    /// Returns the number of outcomes installed — the transition count.
    fn install_expansion<S: JointSnapshot>(
        &mut self,
        run: &mut SearchRun,
        tree: &mut Tree<S>,
        node_id: NodeId,
        samples: &[(usize, usize, u64)],
        successors: Vec<S>,
        leaf_values: &[f64],
    ) -> u32 {
        assert_eq!(samples.len(), successors.len(), "one successor per sample");
        assert_eq!(samples.len(), leaf_values.len(), "one value per sample");
        let transitions = u32::try_from(successors.len()).expect("transition count fits u32");
        // Order-preserving dedup, mirroring Python's dict.fromkeys.
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for &(player, enemy, _) in samples {
            if !pairs.contains(&(player, enemy)) {
                pairs.push((player, enemy));
            }
        }
        let node = tree.node_mut(node_id);
        let parent_potential = node.snapshot.potential();
        for ((&(player, enemy, _), successor), &leaf_value) in
            samples.iter().zip(successors).zip(leaf_values)
        {
            let (tactical_delta, shaped) = if successor.terminal_value().is_some() {
                (0.0, leaf_value)
            } else {
                let delta = successor.potential() - parent_potential;
                (delta, (leaf_value + delta).clamp(-1.0, 1.0))
            };
            node.push_outcome(
                player,
                enemy,
                Outcome {
                    snapshot: successor,
                    tactical_delta,
                    leaf_value,
                },
            );
            node.record_value(player, enemy, shaped);
        }
        node.expanded = true;
        run.expanded_nodes += 1;
        let possible = node.player_legal.len() * node.enemy_legal.len();
        run.possible_joint_pairs += u32::try_from(possible).expect("pair count fits u32");
        run.sampled_joint_pairs += u32::try_from(pairs.len()).expect("pair count fits u32");
        run.chance_outcomes += transitions;
        solve_node(node, self.config.regret_iterations_per_update);
        transitions
    }

    /// The deep/shallow routing predicate chain (`_choose_adaptive_depth`),
    /// in Python's exact order. The budget stream is drawn at most once,
    /// and only when every earlier predicate falls through.
    fn choose_adaptive_depth<S>(&mut self, node: &TreeNode<S>, router_score: f64) -> AdaptiveReason {
        if !self.config.adaptive_search {
            return AdaptiveReason::Disabled;
        }
        if self.config.max_depth <= 1 {
            return AdaptiveReason::ConfiguredRootOnly;
        }
        if router_score >= self.config.adaptive_deep_threshold {
            return AdaptiveReason::LearnedRouter;
        }
        if node.online_exploitability >= self.config.adaptive_exploitability_threshold {
            return AdaptiveReason::RootOnlineExploitability;
        }
        let (max_payoff, min_payoff) = legal_payoff_bounds(node);
        let entropy = policy_entropy(&node.player_policy) + policy_entropy(&node.enemy_policy);
        if max_payoff - min_payoff >= self.config.adaptive_payoff_spread_threshold && entropy >= 1.0 {
            return AdaptiveReason::RootPayoffUncertainty;
        }
        if next_f64(&mut self.budget_rng) < self.config.adaptive_force_deep_fraction {
            return AdaptiveReason::ForcedCalibrationSample;
        }
        AdaptiveReason::RouterStableRoot
    }

    /// The assembly part of `_finish_root_frontier`: on the deep path the
    /// final cold equilibrium overwrites the node and becomes the result;
    /// on the shallow path the initial equilibrium is the result. Change
    /// diagnostics are computed on both paths, the deep-search verdict
    /// only on the deep one.
    fn finish_root<S: JointSnapshot>(
        &mut self,
        run: &mut SearchRun,
        tree: &mut Tree<S>,
        node_id: NodeId,
        transitions: u32,
        options: SearchOptions,
    ) -> SearchResult {
        let simulations = 0u32;
        let converged = false;
        let (initial_player, initial_enemy, initial_value) = {
            let node = tree.node(node_id);
            (
                node.player_policy.clone(),
                node.enemy_policy.clone(),
                node.root_value,
            )
        };
        // The budget-gated descent loop runs here once the descent
        // milestone lands: simulate from the root, count learned
        // simulations, track convergence streaks.
        let (player_policy, enemy_policy, final_value, exploitability) = if run.adaptive_deep_selected
        {
            let node = tree.node_mut(node_id);
            let (player_policy, enemy_policy, value, exploitability) =
                root_equilibrium(node, self.config.regret_iterations);
            node.player_policy.clone_from(&player_policy);
            node.enemy_policy.clone_from(&enemy_policy);
            node.root_value = value;
            node.exploitability = exploitability;
            (player_policy, enemy_policy, value, exploitability)
        } else {
            let node = tree.node(node_id);
            (
                initial_player.clone(),
                initial_enemy.clone(),
                initial_value,
                node.exploitability,
            )
        };
        let node = tree.node(node_id);
        run.deep_policy_change =
            l1_distance(&player_policy, &initial_player) + l1_distance(&enemy_policy, &initial_enemy);
        run.deep_action_changed = argmax_first(&player_policy, &node.player_legal)
            != argmax_first(&initial_player, &node.player_legal)
            || argmax_first(&enemy_policy, &node.enemy_legal)
                != argmax_first(&initial_enemy, &node.enemy_legal);
        let deep_needed = run.deep_action_changed
            || run.deep_policy_change >= 0.25
            || (final_value - initial_value).abs() >= 0.10;
        run.deep_search_needed = if run.adaptive_deep_selected {
            Some(deep_needed)
        } else {
            None
        };
        let (player_action, enemy_action) = if options.sample_actions {
            let player = sample_index(&player_policy, &mut self.selection_rng);
            let enemy = sample_index(&enemy_policy, &mut self.selection_rng);
            (player, enemy)
        } else {
            (
                argmax_first(&player_policy, &node.player_legal),
                argmax_first(&enemy_policy, &node.enemy_legal),
            )
        };
        let (max_payoff, min_payoff) = legal_payoff_bounds(node);
        let diagnostics = self.diagnostics(
            run,
            simulations,
            converged,
            Some(self.root_diagnostics(node)),
        );
        SearchResult {
            player_policy,
            enemy_policy,
            player_action: Some(player_action),
            enemy_action: Some(enemy_action),
            root_value: final_value,
            transitions,
            solver: SolverTag::RmPlusPooledNodeV3,
            exploitability: Some(exploitability),
            payoff_spread: Some(max_payoff - min_payoff),
            payoff_matrix: Some(node.payoff.clone()),
            diagnostics,
            failure: None,
        }
    }

    fn root_diagnostics<S>(&self, node: &TreeNode<S>) -> RootDiagnostics {
        let joint_actions = node.player_legal.len() * node.enemy_legal.len();
        RootDiagnostics {
            joint_actions: u32::try_from(joint_actions).expect("joint actions fit u32"),
            solves: node.solve_count,
            online_exploitability: node.online_exploitability,
            final_exploitability: node.exploitability,
            equilibrium_iterations: self.config.regret_iterations,
        }
    }

    fn diagnostics(
        &self,
        run: &SearchRun,
        simulations: u32,
        converged: bool,
        root: Option<RootDiagnostics>,
    ) -> Diagnostics {
        Diagnostics {
            tree_nodes: run.expanded_nodes,
            tree_simulations: simulations,
            tree_max_depth: run.max_depth_reached,
            chance_outcomes: run.chance_outcomes,
            sampled_joint_coverage: if run.possible_joint_pairs > 0 {
                f64::from(run.sampled_joint_pairs) / f64::from(run.possible_joint_pairs)
            } else {
                0.0
            },
            tree_converged: converged,
            adaptive_deep_selected: run.adaptive_deep_selected,
            adaptive_router_score: run.adaptive_router_score,
            adaptive_reason: run.adaptive_reason,
            deep_policy_change: run.deep_policy_change,
            deep_action_changed: run.deep_action_changed,
            deep_search_needed: run.deep_search_needed,
            root,
        }
    }

    /// The divergence fallback (`_fallback`): zeroed policies, no actions,
    /// the prior value, and diagnostics without a root section.
    fn fallback(
        &self,
        run: &SearchRun,
        action_count: usize,
        prior_value: f64,
        transitions: u32,
        simulations: u32,
        divergence: Divergence,
    ) -> SearchResult {
        SearchResult {
            player_policy: vec![0.0; action_count],
            enemy_policy: vec![0.0; action_count],
            player_action: None,
            enemy_action: None,
            root_value: prior_value,
            transitions,
            solver: SolverTag::DivergenceFallbackV1,
            exploitability: None,
            payoff_spread: None,
            payoff_matrix: None,
            diagnostics: self.diagnostics(run, simulations, false, None),
            failure: Some(divergence),
        }
    }
}

/// The cold equilibrium over a node's accumulated matrix
/// (`_root_equilibrium`).
fn root_equilibrium<S>(node: &TreeNode<S>, iterations: u32) -> (Vec<f64>, Vec<f64>, f64, f64) {
    solve_zero_sum_regret(
        &node.payoff,
        node.action_count(),
        &node.player_priors,
        &node.enemy_priors,
        &node.player_legal,
        &node.enemy_legal,
        iterations,
    )
}

/// Max and min payoff over the node's legal joint cells.
fn legal_payoff_bounds<S>(node: &TreeNode<S>) -> (f64, f64) {
    let mut max_payoff = f64::NEG_INFINITY;
    let mut min_payoff = f64::INFINITY;
    for &player in &node.player_legal {
        for &enemy in &node.enemy_legal {
            let value = node.payoff_at(player, enemy);
            max_payoff = max_payoff.max(value);
            min_payoff = min_payoff.min(value);
        }
    }
    (max_payoff, min_payoff)
}

/// Elementwise L1 distance between two equal-length policy vectors.
fn l1_distance(left: &[f64], right: &[f64]) -> f64 {
    debug_assert_eq!(left.len(), right.len(), "policies must be comparable");
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .sum()
}
