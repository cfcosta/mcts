//! Toy providers and evaluators for the joint-search suites.
//!
//! Everything here is deliberately tiny and deterministic: matrix games
//! whose equilibria are known in closed form, wrappers that script
//! divergence or record the exact step requests, and evaluators with
//! fixed priors so legal-action ordering is under test control.

use mcts_rs::joint::{Divergence, Evaluation, Evaluator, JointSnapshot, TransitionProvider};

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
