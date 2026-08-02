//! The caller-supplied surface of the joint search: positions, the seeded
//! transition function, and the prior/value evaluator. Together these
//! replace the Python search's native engine handles and network calls.

use std::fmt;

/// A game position handed to the search.
///
/// Values and potentials are always from the player's perspective in
/// [-1, 1] (the search clamps shaped values back into that range).
pub trait JointSnapshot {
    /// Stable identity used to pool tree nodes: two snapshots with the
    /// same id are treated as the same position, exactly as the Python
    /// search keyed nodes by native handle.
    fn id(&self) -> u64;
    /// Bitmask of legal player actions — bit `i` set means action `i` is
    /// legal. Must be non-zero for non-terminal positions.
    fn player_mask(&self) -> u64;
    /// Bitmask of legal enemy actions, same encoding.
    fn enemy_mask(&self) -> u64;
    /// `Some(final value)` when the position is terminal.
    fn terminal_value(&self) -> Option<f64>;
    /// Dense shaping potential: a transition's leaf value is shifted by
    /// the potential difference across it. Defaults to no shaping.
    fn potential(&self) -> f64 {
        0.0
    }
}

/// Advances positions through joint actions under seeded chance.
///
/// The same `(parent, player_action, enemy_action, chance_seed)` request
/// must always produce the same successor — expansion deliberately reuses
/// one seed across every pair at the same sample index (common random
/// numbers), and reproducibility of the whole search rests on the
/// provider honoring the seed.
pub trait TransitionProvider {
    type Snapshot: JointSnapshot;

    /// Steps `parent` by a joint action, or reports semantic divergence.
    ///
    /// Divergence is the one in-band failure of the search: it aborts the
    /// current search and surfaces as `SearchResult::failure` rather than
    /// a panic, mirroring the Python `_TreeDivergence` control flow.
    fn step(
        &mut self,
        parent: &Self::Snapshot,
        player_action: usize,
        enemy_action: usize,
        chance_seed: u64,
    ) -> Result<Self::Snapshot, Divergence>;
}

/// Priors and value produced by evaluating one snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct Evaluation {
    /// Player action priors, `action_count` long (need not be normalized;
    /// the search renormalizes over the legal subset).
    pub player_priors: Vec<f64>,
    /// Enemy action priors, same length contract.
    pub enemy_priors: Vec<f64>,
    /// Position value from the player's perspective in [-1, 1].
    pub value: f64,
}

/// The network stand-in: maps snapshots to priors and values.
pub trait Evaluator<S: JointSnapshot> {
    /// Size of both action spaces; fixed for the lifetime of a search and
    /// at most 64 so legality fits the snapshot masks.
    fn action_count(&self) -> usize;
    /// Full evaluation, used wherever the search needs priors (node
    /// creation).
    fn evaluate(&mut self, snapshot: &S) -> Evaluation;
    /// Value-only evaluation for leaf outcomes. The default derives it
    /// from [`evaluate`](Evaluator::evaluate); implementations with a
    /// cheaper value head can override it.
    fn leaf_value(&mut self, snapshot: &S) -> f64 {
        self.evaluate(snapshot).value
    }
}

/// Provider-reported semantic divergence.
///
/// Python collapsed this to the fixed failure string
/// `"search-semantic-divergence"`; here the provider's reason is carried
/// through to `SearchResult::failure`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Divergence {
    pub reason: String,
}

impl Divergence {
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

impl fmt::Display for Divergence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "semantic divergence: {}", self.reason)
    }
}

impl std::error::Error for Divergence {}
