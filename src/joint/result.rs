//! The search's output surface: the result struct, its diagnostics, and
//! the tag enums whose `as_str` forms reproduce the Python search's
//! diagnostic strings verbatim.

use std::fmt;

use super::traits::Divergence;

/// Which solver produced a result — the Python `SearchResult.solver` tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverTag {
    /// The full RM+ matrix-tree search completed.
    RmPlusPooledNodeV3,
    /// The provider diverged and the result is the prior-based fallback.
    DivergenceFallbackV1,
}

impl SolverTag {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RmPlusPooledNodeV3 => "simultaneous-matrix-tree-rmplus-pooled-node-v3",
            Self::DivergenceFallbackV1 => "rmplus-unavailable-on-semantic-divergence-v1",
        }
    }
}

impl fmt::Display for SolverTag {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Why the adaptive router chose deep or shallow search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveReason {
    /// Adaptive routing is off; every search is deep.
    Disabled,
    /// `max_depth <= 1` leaves nothing to deepen.
    ConfiguredRootOnly,
    /// The learned router score cleared its threshold.
    LearnedRouter,
    /// The root's online exploitability cleared its threshold.
    RootOnlineExploitability,
    /// Wide payoff spread with enough policy entropy.
    RootPayoffUncertainty,
    /// A random calibration sample forced a deep search.
    ForcedCalibrationSample,
    /// No predicate fired; the root looked stable.
    RouterStableRoot,
}

impl AdaptiveReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "adaptive-disabled",
            Self::ConfiguredRootOnly => "configured-root-only",
            Self::LearnedRouter => "learned-router",
            Self::RootOnlineExploitability => "root-online-exploitability",
            Self::RootPayoffUncertainty => "root-payoff-uncertainty",
            Self::ForcedCalibrationSample => "forced-calibration-sample",
            Self::RouterStableRoot => "router-stable-root",
        }
    }

    /// Whether this reason routes to a deep search.
    pub fn is_deep(self) -> bool {
        !matches!(self, Self::ConfiguredRootOnly | Self::RouterStableRoot)
    }
}

impl fmt::Display for AdaptiveReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Per-search knobs that are not configuration: whether to sample the
/// final actions (vs argmax) and the learned router score for adaptive
/// routing (Python's pooled budget head; 1.0 — always deep-eligible —
/// when absent).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchOptions {
    pub sample_actions: bool,
    pub router_score: f64,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            sample_actions: true,
            router_score: 1.0,
        }
    }
}

/// Root-level diagnostics, present when the search built a root node
/// (absent on divergence fallback).
#[derive(Debug, Clone, PartialEq)]
pub struct RootDiagnostics {
    /// |player legal| × |enemy legal| at the root.
    pub joint_actions: u32,
    /// Total warm RM+ iterations applied to the root node.
    pub solves: u32,
    /// Exploitability of the first warm solve, before the equilibrium.
    pub online_exploitability: f64,
    /// Exploitability of the root's final policy.
    pub final_exploitability: f64,
    /// Iterations of the cold equilibrium solve (`regret_iterations`).
    pub equilibrium_iterations: u32,
}

/// The Python diagnostics dict as a typed struct. Pipeline-side entries
/// (inference batching, caching, pooling waves) have no equivalent here
/// and are dropped.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostics {
    /// Nodes expanded (Python `tree_nodes`).
    pub tree_nodes: u32,
    /// Descent simulations that learned something.
    pub tree_simulations: u32,
    /// Deepest node depth reached.
    pub tree_max_depth: u32,
    /// Chance outcomes materialized tree-wide.
    pub chance_outcomes: u32,
    /// Sampled joint pairs / possible joint pairs across expanded nodes.
    pub sampled_joint_coverage: f64,
    /// Whether the descent loop stopped on convergence.
    pub tree_converged: bool,
    /// Whether the router selected a deep search.
    pub adaptive_deep_selected: bool,
    /// The router score the search was given.
    pub adaptive_router_score: f64,
    /// Which predicate decided the routing.
    pub adaptive_reason: AdaptiveReason,
    /// L1 distance between final and initial root policies (both sides).
    pub deep_policy_change: f64,
    /// Whether the argmax action changed on either side.
    pub deep_action_changed: bool,
    /// Deep-search verdict for router training; `None` when the search
    /// stayed shallow (Python's -1 sentinel).
    pub deep_search_needed: Option<bool>,
    pub root: Option<RootDiagnostics>,
}

/// The search's answer, mirroring the Python `SearchResult` (with `nodes`
/// renamed `transitions` — it counts provider `step` calls).
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// Final player policy over the full action space (zeroed on failure).
    pub player_policy: Vec<f64>,
    pub enemy_policy: Vec<f64>,
    /// Chosen actions; `None` on failure.
    pub player_action: Option<usize>,
    pub enemy_action: Option<usize>,
    pub root_value: f64,
    /// Provider transitions consumed (Python `nodes`).
    pub transitions: u32,
    pub solver: SolverTag,
    /// Exploitability of the final root policy; `None` on failure.
    pub exploitability: Option<f64>,
    /// max - min over the root's legal payoff cells; `None` on failure.
    pub payoff_spread: Option<f64>,
    /// The root payoff matrix, `action_count`² row-major; `None` on
    /// failure.
    pub payoff_matrix: Option<Vec<f64>>,
    pub diagnostics: Diagnostics,
    /// The provider's divergence when the search fell back, else `None`.
    pub failure: Option<Divergence>,
}
