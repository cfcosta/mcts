//! Goofspiel (the Game of Pure Strategy) played by the joint
//! simultaneous-move search.
//!
//! Each player holds bid cards valued `1..=n`. Every turn a prize card
//! (same values) is face-up; both players secretly bid one card, the
//! higher bid wins the prize's points, and a tied bid discards it. The
//! next prize is then revealed at random from the undealt deck. When the
//! hands run out, the higher score wins.
//!
//! The game is the canonical benchmark for simultaneous-move search:
//! optimal play must randomize bids, so the regret-matching equilibrium
//! policies are genuinely mixed. It exercises every part of the `joint`
//! module — seeded chance (the prize reveal), shrinking legal masks (the
//! hands), and potential shaping (the running score lead).
//!
//! Run with `cargo run --example goofspiel`. The rules are pinned by the
//! hegel property suite in `tests/goofspiel_properties.rs`, which
//! includes this file directly.

use mcts_rs::joint::rng::{next_index, SplitMix64};
use mcts_rs::joint::{Divergence, Evaluation, Evaluator, JointSnapshot, TransitionProvider};

/// One Goofspiel position. Action `i` bids the card valued `i + 1`;
/// hand and deck bitmasks use the same encoding (bit `i` is value
/// `i + 1`). `upcard` is the face-up prize value, `0` only once the
/// game is over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoofState {
    pub n: u8,
    pub player_hand: u16,
    pub enemy_hand: u16,
    pub remaining: u16,
    pub upcard: u8,
    pub player_score: u8,
    pub enemy_score: u8,
}

impl GoofState {
    /// A fresh `n`-card game (`1..=13`) with `first_upcard` already
    /// revealed; the other prizes stay face-down in the deck.
    pub fn new(n: u8, first_upcard: u8) -> Self {
        assert!((1..=13).contains(&n), "goofspiel supports 1..=13 cards");
        assert!(
            (1..=n).contains(&first_upcard),
            "the first upcard must be one of the prizes"
        );
        let full = (1u16 << n) - 1;
        Self {
            n,
            player_hand: full,
            enemy_hand: full,
            remaining: full & !(1 << (first_upcard - 1)),
            upcard: first_upcard,
            player_score: 0,
            enemy_score: 0,
        }
    }

    /// Sum of every prize in the game, the score normalizer.
    pub fn total_points(&self) -> f64 {
        f64::from(u16::from(self.n) * (u16::from(self.n) + 1) / 2)
    }

    /// Normalized score lead in [-1, 1] from the player's perspective.
    pub fn score_lead(&self) -> f64 {
        (f64::from(self.player_score) - f64::from(self.enemy_score)) / self.total_points()
    }
}

impl JointSnapshot for GoofState {
    fn id(&self) -> u64 {
        // n <= 13 keeps every field inside its window: three 13-bit
        // masks, a 4-bit upcard, and two 7-bit scores (at most 91).
        u64::from(self.player_hand)
            | u64::from(self.enemy_hand) << 13
            | u64::from(self.remaining) << 26
            | u64::from(self.upcard) << 39
            | u64::from(self.player_score) << 43
            | u64::from(self.enemy_score) << 50
    }

    fn player_mask(&self) -> u64 {
        u64::from(self.player_hand)
    }

    fn enemy_mask(&self) -> u64 {
        u64::from(self.enemy_hand)
    }

    fn terminal_value(&self) -> Option<f64> {
        (self.player_hand == 0).then(|| self.score_lead())
    }

    fn potential(&self) -> f64 {
        self.score_lead()
    }
}

/// The `index`-th (0-based, ascending) card value present in `mask`.
fn nth_card(mask: u16, index: usize) -> u8 {
    let mut seen = 0;
    for bit in 0u8..13 {
        if mask & (1 << bit) != 0 {
            if seen == index {
                return bit + 1;
            }
            seen += 1;
        }
    }
    unreachable!("card index out of range for the mask")
}

/// Steps Goofspiel positions: resolves the simultaneous bids and reveals
/// the next prize from the chance seed. Pure rules — never diverges.
pub struct GoofProvider;

impl TransitionProvider for GoofProvider {
    type Snapshot = GoofState;

    fn step(
        &mut self,
        parent: &GoofState,
        player_action: usize,
        enemy_action: usize,
        chance_seed: u64,
    ) -> Result<GoofState, Divergence> {
        assert!(parent.player_hand != 0, "cannot step a finished game");
        let player_bit = 1u16 << player_action;
        let enemy_bit = 1u16 << enemy_action;
        assert!(
            parent.player_hand & player_bit != 0,
            "the player bid must come from the hand"
        );
        assert!(
            parent.enemy_hand & enemy_bit != 0,
            "the enemy bid must come from the hand"
        );

        let mut next = *parent;
        next.player_hand &= !player_bit;
        next.enemy_hand &= !enemy_bit;
        if player_action > enemy_action {
            next.player_score += parent.upcard;
        } else if enemy_action > player_action {
            next.enemy_score += parent.upcard;
        }
        // A tied bid discards the prize.

        if next.remaining == 0 {
            next.upcard = 0;
        } else {
            let count = next.remaining.count_ones() as usize;
            let pick = next_index(&mut SplitMix64::new(chance_seed), count);
            let card = nth_card(next.remaining, pick);
            next.remaining &= !(1u16 << (card - 1));
            next.upcard = card;
        }
        Ok(next)
    }
}

/// Total point value of the cards in a mask.
fn card_points(mask: u16) -> f64 {
    let mut sum = 0u32;
    for bit in 0u8..13 {
        if mask & (1 << bit) != 0 {
            sum += u32::from(bit) + 1;
        }
    }
    f64::from(sum)
}

/// Heuristic network stand-in: bids near the prize value get higher
/// priors, and the value estimate blends the realized score lead with
/// the undealt prizes projected onto relative hand strength — a leaf
/// that spent a strong card on a weak prize is worth less, which is
/// what pushes the equilibrium bids away from "always bid highest"
/// and into genuinely mixed strategies.
pub struct ClosenessEvaluator {
    pub n: u8,
}

impl Evaluator<GoofState> for ClosenessEvaluator {
    fn action_count(&self) -> usize {
        usize::from(self.n)
    }

    fn evaluate(&mut self, snapshot: &GoofState) -> Evaluation {
        let priors = |hand: u16| {
            (0u8..self.n)
                .map(|action| {
                    if hand & (1u16 << action) == 0 {
                        0.0
                    } else {
                        let card = f64::from(action + 1);
                        1.0 / (1.0 + (card - f64::from(snapshot.upcard)).abs())
                    }
                })
                .collect()
        };
        let undealt = f64::from(snapshot.upcard) + card_points(snapshot.remaining);
        let strength = card_points(snapshot.player_hand) + card_points(snapshot.enemy_hand);
        let edge = card_points(snapshot.player_hand) - card_points(snapshot.enemy_hand);
        let projected = if strength > 0.0 {
            undealt * edge / strength
        } else {
            0.0
        };
        let realized = f64::from(snapshot.player_score) - f64::from(snapshot.enemy_score);
        Evaluation {
            player_priors: priors(snapshot.player_hand),
            enemy_priors: priors(snapshot.enemy_hand),
            value: (realized + projected) / snapshot.total_points(),
        }
    }
}

/// Renders a hand with the search's mixed bid probabilities.
#[cfg(not(test))]
fn describe(hand: u16, policy: &[f64]) -> String {
    (0u8..16)
        .filter(|bit| hand & (1u16 << bit) != 0)
        .map(|bit| format!("{}:{:.0}%", bit + 1, policy[usize::from(bit)] * 100.0))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(not(test))]
fn main() {
    use mcts_rs::joint::{JointSearchConfig, SearchOptions, SimultaneousTreeSearch};
    use rand::RngCore;

    const CARDS: u8 = 6;
    const SEED: u64 = 0x600F_5D1E;

    let config = JointSearchConfig {
        max_depth: 3,
        expansion_budget: 160,
        minimum_expansion_budget: 96,
        ..JointSearchConfig::default()
    };
    let mut engine = SimultaneousTreeSearch::new(config, SEED);
    let mut provider = GoofProvider;
    let mut evaluator = ClosenessEvaluator { n: CARDS };
    // The real game's chance stream, separate from the search's streams.
    let mut chance = SplitMix64::new(SEED ^ 0x9E37_79B9_7F4A_7C15);

    let first = nth_card(
        (1u16 << CARDS) - 1,
        next_index(&mut chance, usize::from(CARDS)),
    );
    let mut state = GoofState::new(CARDS, first);
    println!("goofspiel 1..={CARDS}, both sides played by the joint search (seed {SEED:#x})");

    for turn in 1..=CARDS {
        let result = engine.search(
            &mut provider,
            &mut evaluator,
            state,
            SearchOptions {
                sample_actions: true,
                router_score: 1.0,
            },
        );
        let player = result
            .player_action
            .expect("a successful search picks a bid");
        let enemy = result
            .enemy_action
            .expect("a successful search picks a bid");
        println!(
            "turn {turn}: prize {:>2} value {:+.3}\n  player bids {:>2} of [{}]\n  enemy  bids {:>2} of [{}]",
            state.upcard,
            result.root_value,
            player + 1,
            describe(state.player_hand, &result.player_policy),
            enemy + 1,
            describe(state.enemy_hand, &result.enemy_policy),
        );
        state = provider
            .step(&state, player, enemy, chance.next_u64())
            .expect("goofspiel rules never diverge");
        println!("  score {} - {}", state.player_score, state.enemy_score);
    }

    let lead = state.terminal_value().expect("all cards have been bid");
    let verdict = match lead.partial_cmp(&0.0) {
        Some(std::cmp::Ordering::Greater) => "the player wins",
        Some(std::cmp::Ordering::Less) => "the enemy wins",
        _ => "a draw",
    };
    println!(
        "final score {} - {}: {verdict}",
        state.player_score, state.enemy_score
    );
}
