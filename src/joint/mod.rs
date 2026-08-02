//! Simultaneous-move regret-matching tree search.
//!
//! A port of the Pokémon battle pipeline's joint search: both players act
//! at once, every tree node carries a dense payoff matrix over joint
//! (player, enemy) actions filled by chance-sampled outcomes, nodes are
//! solved with warm-started regret matching+, and the root policy comes
//! from a cold RM+ equilibrium over the accumulated root matrix.
//!
//! This module is a sibling of the classic [`crate::Mcts`] UCT search —
//! the two share nothing: joint actions, priors, seeded chance, and
//! matrix solves have no representation in the [`crate::State`] trait.
//!
//! # Deviations from the Python implementation
//!
//! The port is algorithm-equivalent, not bit-exact. Every equation,
//! control-flow branch, draw order, and RNG stream assignment matches the
//! Python source, but the following differ deliberately:
//!
//! - **RNG.** Python uses three `random.Random` (Mersenne Twister)
//!   instances; this port derives matching [`SplitMix64`] streams from
//!   one seed. Seeds therefore do not reproduce Python runs. The draw
//!   semantics are frozen in [`rng`]: floats take the top 53 bits of one
//!   `next_u64`, and bounded indices use a widening multiply — neither
//!   matches CPython's `random()` or `randrange` internals.
//! - **Chance seeds** are full `u64` draws; Python uses
//!   `getrandbits(63)`.
//! - **Naming.** The Python result field `nodes` is
//!   [`SearchResult::transitions`](result::SearchResult::transitions),
//!   which is what it counts.
//! - **Config.** `inference_batch_size` (a batching concern of the
//!   subprocess protocol) and `redundant_action_prior_scale` (pre-search
//!   prior shaping done by the caller) are dropped.
//! - **Diagnostics.** Node-pool and evaluation-cache hit counters are
//!   dropped: this engine neither pools nor caches.
//! - **Divergence accounting.** On a root-install divergence,
//!   `transitions` counts the steps actually attempted including the
//!   failing one; Python reports the whole pooled batch. A mid-descent
//!   divergence carries the provider's reason text (Python raises fixed
//!   messages) and discards the in-flight simulation's cost, exactly as
//!   Python does.
//! - **Errors.** Caller contract violations (invalid config, terminal
//!   root, empty legal masks, wrong prior lengths) panic; only provider
//!   divergence is reported in-band via
//!   [`SearchResult::failure`](result::SearchResult::failure).
//!
//! # Opt-in extensions
//!
//! Beyond the port, the config carries extensions that default to off
//! (the defaults reproduce the Python behavior exactly) and are grounded
//! in the MCTS literature:
//!
//! - **Prior-mass action pruning**
//!   ([`prior_mass_cutoff`](config::JointSearchConfig::prior_mass_cutoff) +
//!   [`minimum_actions_per_side`](config::JointSearchConfig::minimum_actions_per_side)):
//!   every node's legal lists keep only the highest-prior prefix holding
//!   the configured share of raw prior mass. RM+ solve cost grows with
//!   the payoff-matrix area, and policy-guided action reduction is the
//!   standard lever against joint-action blowup (Świechowski et al.,
//!   arXiv:2103.04931 — "Action Reduction"); production AlphaZero-style
//!   engines restrict simultaneous-move nodes to the top ~99.5% of
//!   cumulative prior mass.
//! - **Seeded Dirichlet root noise**
//!   ([`root_noise`](config::JointSearchConfig::root_noise)): the root
//!   priors are blended with `(1 − ε)·prior + ε·Dirichlet(α)` before the
//!   root node is built, with `α = alpha_scale / |legal|` per side —
//!   AlphaZero's root exploration noise (Silver et al.,
//!   arXiv:1712.01815), which keeps every root action explorable even
//!   when the evaluator's priors dismiss it. The draws come from a
//!   dedicated fourth rng stream appended after the ported three, so the
//!   selection/chance/budget traces of any seed are unchanged whether or
//!   not noise is enabled.

pub mod config;
pub mod node;
pub mod noise;
pub mod result;
pub mod rng;
pub mod search;
pub mod solver;
pub mod traits;

pub use config::{ConfigError, JointSearchConfig, RootNoise};
pub use node::{legal_from_priors, truncate_to_prior_mass, NodeId, Outcome, Tree, TreeNode};
pub use noise::{apply_root_noise, sample_dirichlet};
pub use result::{
    AdaptiveReason, Diagnostics, RootDiagnostics, SearchOptions, SearchResult, SolverTag,
};
pub use rng::SplitMix64;
pub use search::SimultaneousTreeSearch;
pub use solver::{
    argmax_first, average_policy, chance_resample_probability, expansion_pairs, mixed_policy,
    normalized_prior, policy_entropy, sample_index, solve_node, solve_zero_sum_regret,
};
pub use traits::{Divergence, Evaluation, Evaluator, JointSnapshot, TransitionProvider};
