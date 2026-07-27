//! Reference game implementations of the [`State`](crate::State) trait.
//!
//! These are used by the examples, the integration test suite, and the
//! benchmark suite. They are part of the public API so that all three see
//! exactly the same code.

pub mod tic_tac_toe;
pub mod ultimate_tic_tac_toe;

pub use tic_tac_toe::TicTacToe;
pub use ultimate_tic_tac_toe::UltimateTicTacToe;

/// Uniformly samples `[0, upper)` with Lemire's multiply-high method.
/// Bundled games use this small concrete helper instead of monomorphizing the
/// much larger generic `gen_range` machinery into each rollout selector.
pub(crate) fn random_below(upper: u32) -> u32 {
    use rand::RngCore;

    debug_assert!(upper > 0);
    let mut rng = rand::thread_rng();
    loop {
        let product = u64::from(rng.next_u32()) * u64::from(upper);
        let low = product as u32;
        if low < upper {
            let threshold = upper.wrapping_neg() % upper;
            if low < threshold {
                continue;
            }
        }
        return (product >> 32) as u32;
    }
}
