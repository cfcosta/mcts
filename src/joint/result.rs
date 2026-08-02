//! The search's output surface: the result struct, its diagnostics, and
//! the string-tag enums that label how a result was produced.

use std::fmt;

use super::traits::Divergence;

/// Which solver produced a [`SearchResult`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverTag {
    /// The full RM+ matrix-tree search completed.
    RmPlus,
    /// The provider diverged and the result is the prior-based fallback.
    DivergenceFallback,
}

impl SolverTag {
    /// The stable diagnostic string for this tag.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RmPlus => "simultaneous-matrix-tree-rmplus",
            Self::DivergenceFallback => "rmplus-unavailable-on-semantic-divergence",
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
    /// The stable diagnostic string for this reason.
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
/// routing (from an external routing model when one exists; 1.0 —
/// always deep-eligible — when absent).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchOptions {
    /// Sample the final actions from the policies instead of argmax.
    pub sample_actions: bool,
    /// The learned router score consumed by adaptive routing.
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
    /// Iterations the last cold equilibrium solve actually ran: the
    /// full `regret_iterations`, or the checkpoint where the opt-in
    /// `equilibrium_tolerance` stopped it early.
    pub equilibrium_iterations: u32,
}

/// Tree-level counters and routing telemetry for one search.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostics {
    /// Nodes expanded.
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
    /// stayed shallow.
    pub deep_search_needed: Option<bool>,
    /// Root-node diagnostics; `None` on divergence fallback.
    pub root: Option<RootDiagnostics>,
}

/// The search's answer: final policies, chosen actions, and diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// Final player policy over the full action space (zeroed on failure).
    pub player_policy: Vec<f64>,
    /// Final enemy policy over the full action space (zeroed on failure).
    pub enemy_policy: Vec<f64>,
    /// Chosen player action; `None` on failure.
    pub player_action: Option<usize>,
    /// Chosen enemy action; `None` on failure.
    pub enemy_action: Option<usize>,
    /// The root's game value in the player's view.
    pub root_value: f64,
    /// Provider `step` transitions consumed.
    pub transitions: u32,
    /// Which solver produced this result.
    pub solver: SolverTag,
    /// Exploitability of the final root policy; `None` on failure.
    pub exploitability: Option<f64>,
    /// max - min over the root's legal payoff cells; `None` on failure.
    pub payoff_spread: Option<f64>,
    /// The root payoff matrix, `action_count`² row-major; `None` on
    /// failure.
    pub payoff_matrix: Option<Vec<f64>>,
    /// Tree-level counters and routing telemetry.
    pub diagnostics: Diagnostics,
    /// The provider's divergence when the search fell back, else `None`.
    pub failure: Option<Divergence>,
}
