//! Reference game implementations of the [`State`](crate::State) trait.
//!
//! These are used by the examples, the integration test suite, and the
//! benchmark suite. They are part of the public API so that all three see
//! exactly the same code.

pub mod tic_tac_toe;
pub mod ultimate_tic_tac_toe;

pub use tic_tac_toe::TicTacToe;
pub use ultimate_tic_tac_toe::UltimateTicTacToe;

const BIT_COUNT: [u8; 512] = {
    let mut table = [0; 512];
    let mut bits = 1;
    while bits < table.len() {
        table[bits] = table[bits >> 1] + (bits & 1) as u8;
        bits += 1;
    }
    table
};

#[inline]
pub(crate) fn bit_count(bits: u16) -> u32 {
    u32::from(BIT_COUNT[bits as usize])
}

const NTH_EMPTY_CELL: [[u8; 9]; 512] = {
    let mut table = [[0; 9]; 512];
    let mut bits = 1;
    while bits < table.len() {
        let mut remaining = bits;
        let mut rank = 0;
        while remaining != 0 {
            let position = remaining.trailing_zeros() as u8;
            table[bits][rank] = (position / 3) << 2 | position % 3;
            remaining &= remaining - 1;
            rank += 1;
        }
        bits += 1;
    }
    table
};

#[inline]
pub(crate) fn nth_empty_cell(bits: u16, n: u32) -> (u8, u8) {
    let cell = NTH_EMPTY_CELL[bits as usize][n as usize];
    (cell >> 2, cell & 3)
}

/// Uniformly samples `[0, upper)` with Lemire's multiply-high method.
/// Bundled games use this small concrete helper instead of monomorphizing the
/// much larger generic `gen_range` machinery into each rollout selector.
pub(crate) fn random_below<R: rand::RngCore + ?Sized>(rng: &mut R, upper: u32) -> u32 {
    debug_assert!(upper > 0);
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
