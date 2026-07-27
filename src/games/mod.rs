//! Reference game implementations of the [`State`](crate::State) trait.
//!
//! These are used by the examples, the integration test suite, and the
//! benchmark suite. They are part of the public API so that all three see
//! exactly the same code.

pub mod tic_tac_toe;
pub mod ultimate_tic_tac_toe;

pub use tic_tac_toe::TicTacToe;
pub use ultimate_tic_tac_toe::UltimateTicTacToe;
