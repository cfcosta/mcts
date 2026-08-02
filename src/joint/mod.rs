//! Simultaneous-move regret-matching tree search.
//!
//! Both players act at once: every tree node carries a dense payoff
//! matrix over joint (player, enemy) actions filled by chance-sampled
//! outcomes, nodes are solved with warm-started regret matching+, and
//! the root policy comes from a cold RM+ equilibrium over the
//! accumulated root matrix. The game rules and the evaluation model
//! stay behind the [`TransitionProvider`] and [`Evaluator`] traits, so
//! any two-player zero-sum simultaneous-move game can plug in.
//!
//! This module is a sibling of the classic [`crate::Mcts`] UCT search —
//! the two share nothing: joint actions, priors, seeded chance, and
//! matrix solves have no representation in the [`crate::State`] trait.
//!
//! # Determinism
//!
//! A seed reproduces a search bit for bit. Three [`SplitMix64`] streams
//! (selection, chance, budget) are derived from the one seed, every
//! draw is served by a fixed stream in a fixed order, and the draw
//! semantics are frozen in [`rng`]: floats take the top 53 bits of one
//! `next_u64`, bounded indices use a widening multiply, and chance
//! seeds are full `u64` draws. Matrix products accumulate sequentially
//! left-to-right. Nothing about the environment — platform, allocator,
//! thread timing — enters any result.
//!
//! # Errors
//!
//! Caller contract violations (invalid config, terminal root, empty
//! legal masks, wrong prior lengths, zero solver iterations) panic;
//! only provider divergence is reported in-band, via
//! [`SearchResult::failure`](result::SearchResult::failure) on the
//! documented fallback result.
//!
//! # Opt-in extensions
//!
//! The config carries extensions that default to off (the defaults are
//! the search's frozen baseline dynamics) and are grounded in the
//! game-search literature:
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
//!   dedicated fourth rng stream appended after the three core streams,
//!   so the selection/chance/budget traces of any seed are unchanged
//!   whether or not noise is enabled.
//! - **Average-strategy node policies**
//!   ([`average_strategy_policies`](config::JointSearchConfig::average_strategy_policies)):
//!   warm node solves install the cumulative time-average strategy
//!   `strategy_sum / solve_count` — the quantity regret matching's
//!   no-regret guarantee actually drives to equilibrium — instead of the
//!   default last iterate, which merely cycles around it (the folk
//!   theorem behind Online Outcome Sampling and CFR-family averaging,
//!   Lisý et al.). Node value and exploitability are recomputed on the
//!   averages exactly as the cold solver does, so a node's first solve
//!   reproduces [`solve_zero_sum_regret`] bitwise; the root's policy
//!   remains the cold root equilibrium either way.
//! - **CFR+-style solves**
//!   ([`cfr_plus_solves`](config::JointSearchConfig::cfr_plus_solves)):
//!   every RM+ solve — warm node solves and the cold root equilibrium —
//!   runs with CFR+'s two accelerations (Tammelin, arXiv:1407.5042):
//!   alternating updates, where the enemy's regrets are computed against
//!   the player strategy already refreshed from this iteration's update,
//!   and linear averaging, where iteration `t` enters the strategy
//!   average with weight `t` (warm nodes continue the weights globally
//!   across batches, normalizing by the triangular
//!   [`strategy_weight_total`]). Both
//!   accelerations are exact-convergence-preserving refinements of
//!   regret matching+ that empirically converge much faster than the
//!   simultaneous uniform-average dynamics, which remain the bitwise
//!   default — the dynamics that solved heads-up limit hold'em
//!   (Bowling et al., Science 2015).
//! - **Early-terminated root equilibria**
//!   ([`equilibrium_tolerance`](config::JointSearchConfig::equilibrium_tolerance)):
//!   the cold root equilibria check their time-average exploitability
//!   every [`EQUILIBRIUM_CHECK_INTERVAL`] iterations and stop at the
//!   first checkpoint at or under the
//!   tolerance — solving to a target exploitability rather than a fixed
//!   iteration count, the stopping rule under which CFR+ was deployed
//!   against heads-up limit hold'em (Tammelin, arXiv:1407.5042;
//!   Bowling et al., Science 2015). Regret matching's exploitability
//!   bound decays as `O(1/√T)`, so surplus iterations past the target
//!   are pure cost. A stopped solve is bit-identical to truncating the
//!   fixed-iteration solve at that checkpoint, and the performed count
//!   is surfaced in
//!   [`RootDiagnostics::equilibrium_iterations`](result::RootDiagnostics::equilibrium_iterations).
//!   `None` always runs the full `regret_iterations`, untouched.

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
    argmax_first, average_policy, average_policy_into, chance_resample_probability, expansion_pairs,
    mixed_policy, mixed_policy_into, normalized_prior, policy_entropy, sample_index, solve_node,
    solve_node_with_scratch, solve_zero_sum_regret, solve_zero_sum_regret_with_tolerance,
    strategy_weight_total, SolveScratch, EQUILIBRIUM_CHECK_INTERVAL,
};
pub use traits::{Divergence, Evaluation, Evaluator, JointSnapshot, TransitionProvider};
