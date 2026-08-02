//! The search driver: root installation, descent, and the equilibrium
//! pipeline.
//!
//! This is the engine half of the port — the single-root path of the
//! Python `search`/`search_many` plus `_expansion_spec`,
//! `_install_expansion`, `_simulate_frontier`, `_choose_adaptive_depth`
//! and `_finish_root_frontier`: evaluate the root, install the full
//! joint grid under common random numbers, warm-solve it, overwrite the
//! node with the cold equilibrium, route deep or shallow, run the
//! budget-gated descent loop on the deep path, and assemble the result.
//! The descent loop stops early when the root's time-average policies
//! hold still long enough to count as converged, or when 64 consecutive
//! simulations fail to learn anything.

use rand::RngCore;

use crate::joint::config::JointSearchConfig;
use crate::joint::node::{NodeId, Outcome, Tree, TreeNode};
use crate::joint::noise::apply_root_noise;
use crate::joint::result::{
    AdaptiveReason, Diagnostics, RootDiagnostics, SearchOptions, SearchResult, SolverTag,
};
use crate::joint::rng::{next_f64, next_index, SplitMix64};
use crate::joint::solver::{
    argmax_first, average_policy, chance_resample_probability, expansion_pairs, mixed_policy,
    policy_entropy, sample_index, solve_node, solve_zero_sum_regret, strategy_weight_total,
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
/// A fourth stream, `noise`, has no Python counterpart: it feeds the
/// opt-in Dirichlet root-noise extension exclusively, so enabling noise
/// never shifts a draw on the ported streams.
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
    noise_rng: R,
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
    /// The evaluator's value of the root, before any search: the value the
    /// divergence fallback reports.
    prior_value: f64,
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
            prior_value: 0.0,
            adaptive_deep_selected: true,
            adaptive_router_score: 1.0,
            adaptive_reason: AdaptiveReason::Disabled,
            deep_policy_change: 0.0,
            deep_action_changed: false,
            deep_search_needed: None,
        }
    }
}

/// The provider and evaluator of one search call, bundled so the descent
/// recursion threads a single handle.
struct DriveContext<'a, P, E> {
    provider: &'a mut P,
    evaluator: &'a mut E,
}

impl SimultaneousTreeSearch<SplitMix64> {
    /// Creates a search with four [`SplitMix64`] streams derived from one
    /// seed. The noise stream is drawn after the three ported streams, so
    /// their per-seed traces are identical to builds that predate it.
    /// Panics when the configuration is invalid.
    pub fn new(config: JointSearchConfig, seed: u64) -> Self {
        let mut seeder = SplitMix64::new(seed);
        let selection_rng = SplitMix64::new(seeder.next_u64());
        let chance_rng = SplitMix64::new(seeder.next_u64());
        let budget_rng = SplitMix64::new(seeder.next_u64());
        let noise_rng = SplitMix64::new(seeder.next_u64());
        Self::with_rngs(config, selection_rng, chance_rng, budget_rng, noise_rng)
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
        noise_rng: R,
    ) -> Self {
        if let Err(error) = config.validate() {
            panic!("invalid search config: {error}");
        }
        Self {
            config,
            selection_rng,
            chance_rng,
            budget_rng,
            noise_rng,
        }
    }

    pub fn config(&self) -> &JointSearchConfig {
        &self.config
    }

    /// Runs a search from `root` and returns the result.
    ///
    /// Snapshots are cloned once when a sampled outcome becomes a child
    /// node, so `Clone` should be cheap — snapshots are typically small
    /// handles into the caller's state.
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
        P::Snapshot: Clone,
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
        P::Snapshot: Clone,
        E: Evaluator<P::Snapshot>,
    {
        let action_count = evaluator.action_count();
        assert!(
            root.terminal_value().is_none(),
            "cannot search a terminal state"
        );
        let mut tree = Tree::new(action_count);
        let mut run = SearchRun::new();
        let mut evaluation = evaluator.evaluate(&root);
        run.prior_value = evaluation.value;
        // Root noise perturbs the evaluation before the node exists, so
        // the stored priors — the source of every legal-list derivation
        // downstream — already carry it.
        if let Some(noise) = self.config.root_noise {
            apply_root_noise(
                &mut evaluation,
                root.player_mask(),
                root.enemy_mask(),
                noise,
                self.config.max_actions_per_side,
                &mut self.noise_rng,
            );
        }
        let root_id = tree.make_node(root, evaluation, &self.config);
        let mut ctx = DriveContext {
            provider,
            evaluator,
        };

        // Root install: the full joint grid, exempt from the expansion
        // budget (the budget gates only the descent loop).
        let transitions = match self.expand_node(&mut run, &mut tree, &mut ctx, root_id, true) {
            Ok(transitions) => transitions,
            Err((divergence, steps)) => {
                let result = self.fallback(&run, action_count, steps, 0, divergence);
                return (result, tree);
            }
        };
        {
            let node = tree.node_mut(root_id);
            node.online_exploitability = node.exploitability;
            let (player_policy, enemy_policy, value, exploitability) = root_equilibrium(
                node,
                self.config.regret_iterations,
                self.config.cfr_plus_solves,
            );
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
        let result = self.finish_root(&mut run, &mut tree, &mut ctx, root_id, transitions, options);
        (result, tree)
    }

    /// Expands a node over its joint grid (`_expand_frontier`): the full
    /// legal grid for the budget-exempt root install, diagonal rotations
    /// for nodes reached by descent. One chance seed per sample index is
    /// drawn from the chance stream and shared across every pair — common
    /// random numbers. On divergence, returns the failing step's error and
    /// the number of step calls made; the root reports that count while
    /// descent discards it, matching Python, which loses in-flight descent
    /// cost but reports the whole pooled root batch (the root-side count
    /// is a documented deviation).
    fn expand_node<P, E>(
        &mut self,
        run: &mut SearchRun,
        tree: &mut Tree<P::Snapshot>,
        ctx: &mut DriveContext<'_, P, E>,
        node_id: NodeId,
        full_matrix: bool,
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
                full_matrix,
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
            match ctx
                .provider
                .step(&tree.node(node_id).snapshot, player, enemy, seed)
            {
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
                None => ctx.evaluator.leaf_value(successor),
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
        solve_node(
            node,
            self.config.regret_iterations_per_update,
            self.config.average_strategy_policies,
            self.config.cfr_plus_solves,
        );
        transitions
    }

    /// The transition cost of expanding a node partially (`_joint_cost`).
    fn joint_cost<S>(&self, node: &TreeNode<S>) -> u32 {
        let pairs = expansion_pairs(
            &node.player_legal,
            &node.enemy_legal,
            false,
            self.config.deeper_joint_rotations,
        );
        u32::try_from(pairs.len()).expect("pair count fits u32")
            * self.config.chance_samples_per_joint
    }

    /// One descent from a node (`_simulate_frontier`): sample a joint
    /// action from the epsilon-mixed policies, take a fresh or reused
    /// chance outcome, resolve its value terminally, at the depth cap, or
    /// through the child node, and when the outcome taught us something,
    /// re-record it and warm-solve the node on the way out. Returns
    /// `(value, cost, learned)`; `cost` counts provider transitions.
    fn simulate<P, E>(
        &mut self,
        run: &mut SearchRun,
        tree: &mut Tree<P::Snapshot>,
        ctx: &mut DriveContext<'_, P, E>,
        node_id: NodeId,
        depth: u32,
        remaining: u32,
    ) -> Result<(f64, u32, bool), Divergence>
    where
        P: TransitionProvider,
        P::Snapshot: Clone,
        E: Evaluator<P::Snapshot>,
    {
        run.max_depth_reached = run.max_depth_reached.max(depth);
        let (player_policy, enemy_policy) = {
            let node = tree.node_mut(node_id);
            node.visits += 1;
            let player = mixed_policy(
                &node.player_policy,
                &node.player_priors,
                &node.player_legal,
                node.visits,
                self.config.exploration,
            );
            let enemy = mixed_policy(
                &node.enemy_policy,
                &node.enemy_priors,
                &node.enemy_legal,
                node.visits,
                self.config.exploration,
            );
            (player, enemy)
        };
        let player_action = sample_index(&player_policy, &mut self.selection_rng);
        let enemy_action = sample_index(&enemy_policy, &mut self.selection_rng);
        let mut cost = 0u32;
        let evidence = u32::try_from(
            tree.node(node_id)
                .outcomes_at(player_action, enemy_action)
                .len(),
        )
        .expect("outcome count fits u32");
        if evidence == 0 && remaining == 0 {
            return Ok((tree.node(node_id).root_value, 0, false));
        }
        // Python's short-circuit order: no resample coin is drawn for an
        // unseen pair or an exhausted budget.
        let fresh = evidence == 0
            || (remaining > 0
                && next_f64(&mut self.selection_rng)
                    < chance_resample_probability(evidence, self.config.chance_resample));
        let outcome_index = if fresh {
            // `_new_chance_outcome_frontier`: one fresh chance seed, one
            // step, one evaluation, appended to the pair's outcome cell.
            let seed = self.chance_rng.next_u64();
            let successor = ctx.provider.step(
                &tree.node(node_id).snapshot,
                player_action,
                enemy_action,
                seed,
            )?;
            let leaf_value = match successor.terminal_value() {
                Some(value) => value,
                None => ctx.evaluator.leaf_value(&successor),
            };
            let node = tree.node_mut(node_id);
            let tactical_delta = if successor.terminal_value().is_some() {
                0.0
            } else {
                successor.potential() - node.snapshot.potential()
            };
            node.push_outcome(
                player_action,
                enemy_action,
                Outcome {
                    snapshot: successor,
                    tactical_delta,
                    leaf_value,
                },
            );
            if evidence == 0 {
                run.sampled_joint_pairs += 1;
            }
            run.chance_outcomes += 1;
            cost = 1;
            evidence as usize
        } else {
            next_index(&mut self.selection_rng, evidence as usize)
        };
        let (terminal, tactical_delta, leaf_value, successor_id) = {
            let outcome = &tree.node(node_id).outcomes_at(player_action, enemy_action)[outcome_index];
            (
                outcome.snapshot.terminal_value(),
                outcome.tactical_delta,
                outcome.leaf_value,
                outcome.snapshot.id(),
            )
        };
        let (value, learned) = if let Some(terminal_value) = terminal {
            (terminal_value, fresh)
        } else if depth + 1 >= self.config.max_depth {
            ((leaf_value + tactical_delta).clamp(-1.0, 1.0), fresh)
        } else {
            let child_id = match tree.node(node_id).children.get(&successor_id).copied() {
                Some(child_id) => child_id,
                None => {
                    let successor = tree.node(node_id).outcomes_at(player_action, enemy_action)
                        [outcome_index]
                        .snapshot
                        .clone();
                    let evaluation = ctx.evaluator.evaluate(&successor);
                    let child_id = tree.make_node(successor, evaluation, &self.config);
                    tree.node_mut(node_id)
                        .children
                        .insert(successor_id, child_id);
                    child_id
                }
            };
            if !tree.node(child_id).expanded {
                let expansion_cost = self.joint_cost(tree.node(child_id));
                if cost + expansion_cost <= remaining {
                    cost += self
                        .expand_node(run, tree, ctx, child_id, false)
                        .map_err(|(divergence, _)| divergence)?;
                    run.max_depth_reached = run.max_depth_reached.max(depth + 1);
                    (
                        (tactical_delta + tree.node(child_id).root_value).clamp(-1.0, 1.0),
                        true,
                    )
                } else {
                    // Budget-starved: fall back to the shaped leaf value.
                    ((leaf_value + tactical_delta).clamp(-1.0, 1.0), fresh)
                }
            } else {
                let (child_value, child_cost, child_learned) =
                    self.simulate(run, tree, ctx, child_id, depth + 1, remaining - cost)?;
                cost += child_cost;
                (
                    (tactical_delta + child_value).clamp(-1.0, 1.0),
                    fresh || child_learned,
                )
            }
        };
        if learned {
            let node = tree.node_mut(node_id);
            node.record_value(player_action, enemy_action, value);
            solve_node(
                node,
                self.config.regret_iterations_per_update,
                self.config.average_strategy_policies,
                self.config.cfr_plus_solves,
            );
        }
        Ok((value, cost, learned))
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

    /// The root driver (`_finish_root_frontier`): on the deep path, run
    /// budget-gated descents from the root, then let a final cold
    /// equilibrium overwrite the node and become the result; on the
    /// shallow path the initial equilibrium is the result untouched.
    ///
    /// The loop stops before the budget when the root's time-average
    /// policies move at most `convergence_tolerance` (L1, both sides) for
    /// `convergence_patience` consecutive learned simulations — but only
    /// once the minimum budget is spent and the descent has reached the
    /// depth cap — or when 64 consecutive simulations learn nothing.
    ///
    /// Change diagnostics are computed on both paths, the deep-search
    /// verdict only on the deep one. A divergence raised mid-descent
    /// abandons the tree's learning and returns the fallback, with the
    /// diverging simulation's in-flight cost excluded from `transitions`
    /// exactly as in Python.
    fn finish_root<P, E>(
        &mut self,
        run: &mut SearchRun,
        tree: &mut Tree<P::Snapshot>,
        ctx: &mut DriveContext<'_, P, E>,
        node_id: NodeId,
        transitions: u32,
        options: SearchOptions,
    ) -> SearchResult
    where
        P: TransitionProvider,
        P::Snapshot: Clone,
        E: Evaluator<P::Snapshot>,
    {
        let mut transitions = transitions;
        let mut simulations = 0u32;
        let mut converged = false;
        let (initial_player, initial_enemy, initial_value) = {
            let node = tree.node(node_id);
            (
                node.player_policy.clone(),
                node.enemy_policy.clone(),
                node.root_value,
            )
        };
        let mut previous_player = initial_player.clone();
        let mut previous_enemy = initial_enemy.clone();
        let mut stable_updates = 0u32;
        let mut attempts_without_learning = 0u32;
        while run.adaptive_deep_selected && transitions < self.config.expansion_budget {
            let remaining = self.config.expansion_budget - transitions;
            match self.simulate(run, tree, ctx, node_id, 0, remaining) {
                Ok((_, cost, learned)) => {
                    transitions += cost;
                    if learned {
                        simulations += 1;
                        attempts_without_learning = 0;
                        let node = tree.node(node_id);
                        let weight_total =
                            strategy_weight_total(self.config.cfr_plus_solves, node.solve_count);
                        let current_player = average_policy(
                            &node.player_strategy_sum,
                            weight_total,
                            &node.player_policy,
                        );
                        let current_enemy = average_policy(
                            &node.enemy_strategy_sum,
                            weight_total,
                            &node.enemy_policy,
                        );
                        let change = l1_distance(&current_player, &previous_player)
                            + l1_distance(&current_enemy, &previous_enemy);
                        stable_updates = if change <= self.config.convergence_tolerance {
                            stable_updates + 1
                        } else {
                            0
                        };
                        previous_player = current_player;
                        previous_enemy = current_enemy;
                        if transitions >= self.config.minimum_expansion_budget
                            && run.max_depth_reached >= self.config.max_depth - 1
                            && stable_updates >= self.config.convergence_patience
                        {
                            converged = true;
                            break;
                        }
                    } else {
                        attempts_without_learning += 1;
                        // Hardcoded in Python: give up on a root whose
                        // reachable frontier has stopped producing updates.
                        if attempts_without_learning >= 64 {
                            break;
                        }
                    }
                }
                Err(divergence) => {
                    let action_count = tree.action_count;
                    return self.fallback(run, action_count, transitions, simulations, divergence);
                }
            }
        }
        let (player_policy, enemy_policy, final_value, exploitability) = if run.adaptive_deep_selected
        {
            let node = tree.node_mut(node_id);
            let (player_policy, enemy_policy, value, exploitability) = root_equilibrium(
                node,
                self.config.regret_iterations,
                self.config.cfr_plus_solves,
            );
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
        transitions: u32,
        simulations: u32,
        divergence: Divergence,
    ) -> SearchResult {
        SearchResult {
            player_policy: vec![0.0; action_count],
            enemy_policy: vec![0.0; action_count],
            player_action: None,
            enemy_action: None,
            root_value: run.prior_value,
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
fn root_equilibrium<S>(
    node: &TreeNode<S>,
    iterations: u32,
    cfr_plus: bool,
) -> (Vec<f64>, Vec<f64>, f64, f64) {
    solve_zero_sum_regret(
        &node.payoff,
        node.action_count(),
        &node.player_priors,
        &node.enemy_priors,
        &node.player_legal,
        &node.enemy_legal,
        iterations,
        cfr_plus,
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
