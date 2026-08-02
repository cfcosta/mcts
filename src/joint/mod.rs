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

pub mod config;
pub mod result;
pub mod rng;
pub mod solver;
pub mod traits;

pub use config::{ConfigError, JointSearchConfig};
pub use result::{
    AdaptiveReason, Diagnostics, RootDiagnostics, SearchOptions, SearchResult, SolverTag,
};
pub use rng::SplitMix64;
pub use solver::{normalized_prior, solve_zero_sum_regret};
pub use traits::{Divergence, Evaluation, Evaluator, JointSnapshot, TransitionProvider};
