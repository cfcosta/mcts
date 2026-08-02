//! Deterministic pseudo-randomness for the joint search.
//!
//! The search promises seed-stable traces across library versions and
//! platforms, so it cannot draw through `rand`'s distribution machinery:
//! `StdRng`/`SmallRng` algorithms and the float/range conversions are
//! documented as version-unstable. Every draw instead flows through
//! [`SplitMix64`] and the two helpers below, which freeze the exact draw
//! semantics — one `next_u64` per float, one per index, no rejection
//! loops.

use rand::{Error, RngCore, SeedableRng};

/// Sebastiano Vigna's SplitMix64: eight bytes of state, a full 2^64
/// period, and a fixed output function that is trivial to reimplement in
/// any language when a trace needs to be reproduced outside this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}

impl RngCore for SplitMix64 {
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut mixed = self.state;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        mixed ^ (mixed >> 31)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut chunks = dest.chunks_exact_mut(8);
        for chunk in chunks.by_ref() {
            chunk.copy_from_slice(&self.next_u64().to_le_bytes());
        }
        let remainder = chunks.into_remainder();
        if !remainder.is_empty() {
            let bytes = self.next_u64().to_le_bytes();
            remainder.copy_from_slice(&bytes[..remainder.len()]);
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl SeedableRng for SplitMix64 {
    type Seed = [u8; 8];

    fn from_seed(seed: Self::Seed) -> Self {
        Self::new(u64::from_le_bytes(seed))
    }

    fn seed_from_u64(state: u64) -> Self {
        Self::new(state)
    }
}

/// A uniform draw in [0, 1) from the top 53 bits of one `next_u64`, the
/// float construction CPython's `random.random` also uses.
pub fn next_f64<R: RngCore + ?Sized>(rng: &mut R) -> f64 {
    (rng.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0 // 2^53
}

/// A uniform index in [0, len) via one 128-bit widening multiply.
///
/// Exactly one `next_u64` per call: seeded traces stay aligned no matter
/// which indices are drawn, unlike rejection sampling.
pub fn next_index<R: RngCore + ?Sized>(rng: &mut R, len: usize) -> usize {
    assert!(len > 0, "cannot draw an index from an empty range");
    ((u128::from(rng.next_u64()) * len as u128) >> 64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference outputs of Vigna's splitmix64.c; any drift here would
    // silently invalidate every seeded trace in the test suite.
    const SEED_ZERO_DRAWS: [u64; 4] = [
        0xE220_A839_7B1D_CDAF,
        0x6E78_9E6A_A1B9_65F4,
        0x06C4_5D18_8009_454F,
        0xF88B_B8A8_724C_81EC,
    ];

    #[test]
    fn matches_reference_streams() {
        let mut rng = SplitMix64::new(0);
        let drawn: Vec<u64> = (0..4).map(|_| rng.next_u64()).collect();
        assert_eq!(drawn, SEED_ZERO_DRAWS);

        let mut rng = SplitMix64::new(42);
        assert_eq!(rng.next_u64(), 0xBDD7_3226_2FEB_6E95);
        assert_eq!(rng.next_u64(), 0x28EF_E333_B266_F103);

        let mut rng = SplitMix64::new(0x0123_4567_89AB_CDEF);
        assert_eq!(rng.next_u64(), 0x157A_3807_A48F_AA9D);
        assert_eq!(rng.next_u64(), 0xD573_529B_34A1_D093);
    }

    #[test]
    fn seeding_paths_agree() {
        let mut from_new = SplitMix64::new(0xDEAD_BEEF);
        let mut from_u64 = SplitMix64::seed_from_u64(0xDEAD_BEEF);
        let mut from_bytes = SplitMix64::from_seed(0xDEAD_BEEF_u64.to_le_bytes());
        let expected = from_new.next_u64();
        assert_eq!(from_u64.next_u64(), expected);
        assert_eq!(from_bytes.next_u64(), expected);
    }

    #[test]
    fn float_draws_are_pinned_and_in_range() {
        let mut rng = SplitMix64::new(0);
        assert_eq!(next_f64(&mut rng), 0.883_310_808_213_642_6);
        for _ in 0..1_000 {
            let value = next_f64(&mut rng);
            assert!((0.0..1.0).contains(&value));
        }
    }

    #[test]
    fn index_draws_are_pinned_and_bounded() {
        let mut rng = SplitMix64::new(0);
        rng.next_u64();
        assert_eq!(next_index(&mut rng, 13), 5);
        let mut seen = [false; 5];
        for _ in 0..1_000 {
            seen[next_index(&mut rng, 5)] = true;
        }
        assert!(seen.iter().all(|&hit| hit));
        assert_eq!(next_index(&mut SplitMix64::new(7), 1), 0);
    }

    #[test]
    #[should_panic(expected = "empty range")]
    fn index_draw_rejects_empty_range() {
        next_index(&mut SplitMix64::new(0), 0);
    }

    #[test]
    fn fill_bytes_matches_word_stream() {
        let mut rng = SplitMix64::new(0);
        let mut buffer = [0u8; 9];
        rng.fill_bytes(&mut buffer);
        assert_eq!(buffer, [175, 205, 29, 123, 57, 168, 32, 226, 244]);
    }
}
