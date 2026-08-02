//! Two seeded tree searches for two-player zero-sum games.
//!
//! - [`Mcts`] is a classic single-threaded UCT search over the
//!   [`State`] trait: arena-allocated nodes, UCB1 selection, uniform
//!   random rollouts.
//! - [`joint`] is a simultaneous-move regret-matching tree search:
//!   both players act at once, every node carries a dense payoff matrix
//!   over joint actions, and policies come from warm-started RM+ solves
//!   with a cold root equilibrium — behind its own provider and
//!   evaluator traits.
//!
//! The two searches are independent: they share no traits, no tree
//! representation, and no randomness machinery.

#![warn(missing_docs)]

pub mod joint;
pub mod mcts;
pub mod state;

pub use bumpalo::Bump;
pub use mcts::Mcts;
pub use state::State;
