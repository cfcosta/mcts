//! Hegel property suite for the Goofspiel example.
//!
//! The game rules are pinned by a differential oracle (an independent
//! shadow bookkeeper replays every turn), an exact role-swap metamorphic
//! relation, and seeded-chance determinism. The joint search is then
//! swept over random reachable positions with the structural invariant
//! checker as the end-to-end oracle, plus two exact anchors: forced
//! last-turn bids and the zero value of fresh symmetric games.

// The game under test lives in the example so `cargo run --example
// goofspiel` stays one self-contained file; including it here keeps a
// single source of truth (its `main` is compiled out under cfg(test)).
#[path = "../examples/goofspiel.rs"]
mod goofspiel;

mod support;

use std::collections::HashSet;

use goofspiel::{ClosenessEvaluator, GoofProvider, GoofState};
use hegel::generators as gs;
use hegel::TestCase;
use mcts_rs::joint::{
    Evaluator, JointSearchConfig, JointSnapshot, SearchOptions, SimultaneousTreeSearch,
    TransitionProvider,
};
use support::joint::assert_joint_tree_invariants;

// ---------------------------------------------------------------------------
// Draw helpers: every input is valid by construction, nothing is rejected.
// ---------------------------------------------------------------------------

fn draw_unit(tc: &TestCase) -> f64 {
    tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0))
}

fn draw_seed(tc: &TestCase) -> u64 {
    tc.draw(gs::integers::<u64>())
}

/// A fresh `n`-card game with a drawn first upcard.
fn draw_fresh_state(tc: &TestCase, n: u8) -> GoofState {
    let first = tc.draw(gs::integers::<u8>().min_value(1).max_value(n));
    GoofState::new(n, first)
}

/// The card values still present in a hand or deck mask.
fn mask_to_cards(mask: u16) -> Vec<u8> {
    (0..16)
        .filter(|bit| mask & (1 << bit) != 0)
        .map(|bit| bit + 1)
        .collect()
}

/// Draws a legal bid (an action index) from a non-empty hand mask.
fn draw_bid(tc: &TestCase, hand: u16) -> usize {
    let cards = mask_to_cards(hand);
    let pick = tc.draw(
        gs::integers::<usize>()
            .min_value(0)
            .max_value(cards.len() - 1),
    );
    usize::from(cards[pick]) - 1
}

/// Plays `turns` random legal joint bids under drawn chance seeds.
fn play_random_turns(
    tc: &TestCase,
    provider: &mut GoofProvider,
    mut state: GoofState,
    turns: u8,
) -> GoofState {
    for _ in 0..turns {
        let player = draw_bid(tc, state.player_hand);
        let enemy = draw_bid(tc, state.enemy_hand);
        let seed = draw_seed(tc);
        state = provider
            .step(&state, player, enemy, seed)
            .expect("goofspiel rules never diverge");
    }
    state
}

/// The role-swapped view of a position (test-side, via the pub fields).
fn mirror(state: &GoofState) -> GoofState {
    GoofState {
        player_hand: state.enemy_hand,
        enemy_hand: state.player_hand,
        player_score: state.enemy_score,
        enemy_score: state.player_score,
        ..*state
    }
}

/// A search config drawn entirely inside `validate()`-safe ranges, kept
/// small so the sweep stays fast on real game trees.
fn draw_search_config(tc: &TestCase) -> JointSearchConfig {
    JointSearchConfig {
        chance_samples_per_joint: tc.draw(gs::integers::<u32>().min_value(1).max_value(2)),
        max_depth: tc.draw(gs::integers::<u32>().min_value(1).max_value(3)),
        regret_iterations: tc.draw(gs::integers::<u32>().min_value(8).max_value(128)),
        max_actions_per_side: tc.draw(gs::integers::<usize>().min_value(1).max_value(6)),
        expansion_budget: tc.draw(gs::integers::<u32>().min_value(1).max_value(48)),
        exploration: draw_unit(tc) * 0.5,
        chance_resample: draw_unit(tc),
        regret_iterations_per_update: tc.draw(gs::integers::<u32>().min_value(1).max_value(16)),
        deeper_joint_rotations: tc.draw(gs::integers::<usize>().min_value(1).max_value(3)),
        minimum_expansion_budget: tc.draw(gs::integers::<u32>().min_value(1).max_value(48)),
        convergence_tolerance: draw_unit(tc) * 0.05,
        convergence_patience: tc.draw(gs::integers::<u32>().min_value(1).max_value(8)),
        adaptive_search: tc.draw(gs::booleans()),
        adaptive_force_deep_fraction: draw_unit(tc),
        ..JointSearchConfig::default()
    }
}

// ---------------------------------------------------------------------------
// Shadow bookkeeper: an independent, naive replay of the rules used as
// the differential oracle. It stores hands and prizes as plain card
// lists and tracks discarded points explicitly.
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct ShadowGame {
    n: u8,
    player_hand: Vec<u8>,
    enemy_hand: Vec<u8>,
    remaining: Vec<u8>,
    upcard: u8,
    player_score: u32,
    enemy_score: u32,
    discarded: u32,
}

impl ShadowGame {
    fn new(n: u8, first_upcard: u8) -> Self {
        Self {
            n,
            player_hand: (1..=n).collect(),
            enemy_hand: (1..=n).collect(),
            remaining: (1..=n).filter(|&card| card != first_upcard).collect(),
            upcard: first_upcard,
            player_score: 0,
            enemy_score: 0,
            discarded: 0,
        }
    }

    /// Replays one turn given the bids and the prize the engine under
    /// test claims to have revealed next (`0` for none).
    fn step(&mut self, player_card: u8, enemy_card: u8, revealed: u8) {
        assert!(
            self.player_hand.contains(&player_card),
            "player bid from hand"
        );
        assert!(self.enemy_hand.contains(&enemy_card), "enemy bid from hand");
        self.player_hand.retain(|&card| card != player_card);
        self.enemy_hand.retain(|&card| card != enemy_card);

        let prize = u32::from(self.upcard);
        if player_card > enemy_card {
            self.player_score += prize;
        } else if enemy_card > player_card {
            self.enemy_score += prize;
        } else {
            self.discarded += prize;
        }

        if self.remaining.is_empty() {
            assert_eq!(revealed, 0, "no prize left to reveal");
        } else {
            assert!(
                self.remaining.contains(&revealed),
                "revealed prize {revealed} must come from the face-down deck {:?}",
                self.remaining
            );
            self.remaining.retain(|&card| card != revealed);
        }
        self.upcard = revealed;
    }

    /// Every point in the game is somewhere: won, discarded, face-up, or
    /// face-down.
    fn assert_conserved(&self) {
        let total = u32::from(self.n) * (u32::from(self.n) + 1) / 2;
        let face_down: u32 = self.remaining.iter().map(|&card| u32::from(card)).sum();
        assert_eq!(
            self.player_score
                + self.enemy_score
                + self.discarded
                + u32::from(self.upcard)
                + face_down,
            total,
            "points conservation"
        );
        assert_eq!(
            self.player_hand.len(),
            self.enemy_hand.len(),
            "hand sizes stay equal"
        );
        let expected_deck = self.player_hand.len().saturating_sub(1);
        assert_eq!(
            self.remaining.len(),
            expected_deck,
            "deck trails the hands by one"
        );
    }

    fn assert_matches(&self, state: &GoofState) {
        assert_eq!(state.n, self.n, "card count");
        assert_eq!(
            mask_to_cards(state.player_hand),
            self.player_hand,
            "player hand"
        );
        assert_eq!(
            mask_to_cards(state.enemy_hand),
            self.enemy_hand,
            "enemy hand"
        );
        assert_eq!(
            mask_to_cards(state.remaining),
            self.remaining,
            "face-down deck"
        );
        assert_eq!(state.upcard, self.upcard, "upcard");
        assert_eq!(
            u32::from(state.player_score),
            self.player_score,
            "player score"
        );
        assert_eq!(
            u32::from(state.enemy_score),
            self.enemy_score,
            "enemy score"
        );
    }
}

// ---------------------------------------------------------------------------
// Rule properties.
// ---------------------------------------------------------------------------

/// Differential oracle: a full random game agrees with the shadow
/// bookkeeper after every turn, points are conserved throughout, and the
/// terminal law holds — `terminal_value` is `None` exactly until the
/// hands empty, then equals the normalized score lead.
#[hegel::test(test_cases = 256)]
fn random_games_match_a_shadow_bookkeeper(tc: TestCase) {
    let n = tc.draw(gs::integers::<u8>().min_value(1).max_value(13));
    let first = tc.draw(gs::integers::<u8>().min_value(1).max_value(n));
    let mut state = GoofState::new(n, first);
    let mut shadow = ShadowGame::new(n, first);
    let mut provider = GoofProvider;
    shadow.assert_matches(&state);
    shadow.assert_conserved();

    for _ in 0..n {
        assert_eq!(
            state.terminal_value(),
            None,
            "live positions are not terminal"
        );
        let player = draw_bid(&tc, state.player_hand);
        let enemy = draw_bid(&tc, state.enemy_hand);
        let seed = draw_seed(&tc);
        state = provider
            .step(&state, player, enemy, seed)
            .expect("goofspiel rules never diverge");
        shadow.step(player as u8 + 1, enemy as u8 + 1, state.upcard);
        shadow.assert_matches(&state);
        shadow.assert_conserved();
    }

    assert_eq!(state.player_hand, 0, "all bid cards spent");
    assert_eq!(state.remaining, 0, "all prizes dealt");
    assert_eq!(state.upcard, 0, "no upcard after the last turn");
    let expected =
        (f64::from(shadow.player_score) - f64::from(shadow.enemy_score)) / state.total_points();
    assert_eq!(
        state.terminal_value(),
        Some(expected),
        "terminal value is the score lead"
    );
    assert!(expected.abs() <= 1.0, "terminal value stays in [-1, 1]");
}

/// Metamorphic relation, exact: swapping the players and their bids
/// yields the mirrored successor under the same chance seed, and
/// mirroring is an involution.
#[hegel::test(test_cases = 256)]
fn swapping_roles_mirrors_every_transition(tc: TestCase) {
    let n = tc.draw(gs::integers::<u8>().min_value(1).max_value(13));
    let mut provider = GoofProvider;
    let fresh = draw_fresh_state(&tc, n);
    let turns = tc.draw(gs::integers::<u8>().min_value(0).max_value(n - 1));
    let state = play_random_turns(&tc, &mut provider, fresh, turns);
    assert_eq!(mirror(&mirror(&state)), state, "mirroring is an involution");

    let player = draw_bid(&tc, state.player_hand);
    let enemy = draw_bid(&tc, state.enemy_hand);
    let seed = draw_seed(&tc);
    let stepped = provider
        .step(&state, player, enemy, seed)
        .expect("goofspiel rules never diverge");
    let swapped = provider
        .step(&mirror(&state), enemy, player, seed)
        .expect("goofspiel rules never diverge");
    assert_eq!(
        swapped,
        mirror(&stepped),
        "role swap commutes with stepping"
    );
}

/// Chance is a pure function of the seed, and whatever it reveals is
/// legal: the same request repeats bitwise, the revealed prize always
/// comes from the face-down deck, and the bid cards leave the hands.
#[hegel::test(test_cases = 256)]
fn chance_reveals_are_seeded_and_legal(tc: TestCase) {
    let n = tc.draw(gs::integers::<u8>().min_value(1).max_value(13));
    let mut provider = GoofProvider;
    let fresh = draw_fresh_state(&tc, n);
    let turns = tc.draw(gs::integers::<u8>().min_value(0).max_value(n - 1));
    let state = play_random_turns(&tc, &mut provider, fresh, turns);

    let player = draw_bid(&tc, state.player_hand);
    let enemy = draw_bid(&tc, state.enemy_hand);
    let seed = draw_seed(&tc);
    let once = provider
        .step(&state, player, enemy, seed)
        .expect("first step");
    let twice = provider
        .step(&state, player, enemy, seed)
        .expect("second step");
    assert_eq!(
        once, twice,
        "identical requests produce identical successors"
    );

    let other = provider
        .step(&state, player, enemy, draw_seed(&tc))
        .expect("step under another seed");
    for successor in [&once, &other] {
        assert_eq!(
            successor.player_hand,
            state.player_hand & !(1 << player),
            "player bid leaves the hand"
        );
        assert_eq!(
            successor.enemy_hand,
            state.enemy_hand & !(1 << enemy),
            "enemy bid leaves the hand"
        );
        if state.remaining == 0 {
            assert_eq!(successor.upcard, 0, "nothing left to reveal");
            assert_eq!(successor.remaining, 0, "deck stays empty");
        } else {
            let bit = 1u16 << (successor.upcard - 1);
            assert_ne!(state.remaining & bit, 0, "revealed prize was face-down");
            assert_eq!(
                successor.remaining,
                state.remaining & !bit,
                "revealed prize left the deck"
            );
        }
    }
}

/// Node pooling keys on `id()`, so positions along a game must never
/// collide — hands shrink every turn, making each state unique.
#[hegel::test(test_cases = 128)]
fn states_along_a_game_never_share_an_id(tc: TestCase) {
    let n = tc.draw(gs::integers::<u8>().min_value(1).max_value(13));
    let mut provider = GoofProvider;
    let mut state = draw_fresh_state(&tc, n);
    let mut ids = HashSet::from([state.id()]);
    for _ in 0..n {
        state = play_random_turns(&tc, &mut provider, state, 1);
        assert!(
            ids.insert(state.id()),
            "state ids must be unique along a game"
        );
    }
    assert_eq!(ids.len(), usize::from(n) + 1);
}

// ---------------------------------------------------------------------------
// Search properties: the joint engine on real Goofspiel positions.
// ---------------------------------------------------------------------------

/// End-to-end structural oracle: random reachable positions, random
/// valid configs — every search must satisfy the full tree invariant
/// checker and never diverge.
#[hegel::test]
fn search_upholds_every_tree_invariant_on_goofspiel(tc: TestCase) {
    let n = tc.draw(gs::integers::<u8>().min_value(2).max_value(4));
    let mut provider = GoofProvider;
    let fresh = draw_fresh_state(&tc, n);
    let turns = tc.draw(gs::integers::<u8>().min_value(0).max_value(n - 1));
    let state = play_random_turns(&tc, &mut provider, fresh, turns);

    let config = draw_search_config(&tc);
    let options = SearchOptions {
        sample_actions: tc.draw(gs::booleans()),
        router_score: draw_unit(&tc),
    };
    let mut engine = SimultaneousTreeSearch::new(config.clone(), draw_seed(&tc));
    let mut evaluator = ClosenessEvaluator { n };
    let (result, tree) = engine.search_with_tree(&mut provider, &mut evaluator, state, options);

    assert!(result.failure.is_none(), "goofspiel never diverges");
    assert_joint_tree_invariants(&tree, &result, &config, "goofspiel sweep");
}

/// Exact anchor: with one card left in each hand the bids are forced, so
/// both equilibrium policies must be point masses on the remaining card
/// and the chosen actions must match it.
#[hegel::test]
fn forced_last_bids_are_point_masses(tc: TestCase) {
    let n = tc.draw(gs::integers::<u8>().min_value(2).max_value(5));
    let mut provider = GoofProvider;
    let fresh = draw_fresh_state(&tc, n);
    let state = play_random_turns(&tc, &mut provider, fresh, n - 1);
    let player_action = state.player_hand.trailing_zeros() as usize;
    let enemy_action = state.enemy_hand.trailing_zeros() as usize;

    let config = JointSearchConfig {
        max_depth: tc.draw(gs::integers::<u32>().min_value(1).max_value(2)),
        expansion_budget: tc.draw(gs::integers::<u32>().min_value(1).max_value(8)),
        minimum_expansion_budget: 1,
        regret_iterations: 64,
        ..JointSearchConfig::default()
    };
    let options = SearchOptions {
        sample_actions: false,
        router_score: 1.0,
    };
    let mut engine = SimultaneousTreeSearch::new(config, draw_seed(&tc));
    let mut evaluator = ClosenessEvaluator { n };
    let result = engine.search(&mut provider, &mut evaluator, state, options);

    for action in 0..usize::from(n) {
        let expected_player = if action == player_action { 1.0 } else { 0.0 };
        let expected_enemy = if action == enemy_action { 1.0 } else { 0.0 };
        assert_eq!(
            result.player_policy[action], expected_player,
            "forced player bid"
        );
        assert_eq!(
            result.enemy_policy[action], expected_enemy,
            "forced enemy bid"
        );
    }
    assert_eq!(result.player_action, Some(player_action));
    assert_eq!(result.enemy_action, Some(enemy_action));
}

/// With equal scores and equally sized hands, the side holding the
/// stronger bid cards must be favored by the evaluator: the undealt
/// prizes are still up for grabs and only hand strength can claim them.
/// Also pins the exact evaluation symmetries the fresh-game test leans
/// on: value antisymmetry and prior swapping under a role mirror.
#[hegel::test(test_cases = 256)]
fn stronger_hands_are_worth_more(tc: TestCase) {
    let n = tc.draw(gs::integers::<u8>().min_value(2).max_value(13));
    // Two distinct swing cards; every other card is shared by both hands.
    let low = tc.draw(gs::integers::<u8>().min_value(0).max_value(n - 2));
    let high = tc.draw(gs::integers::<u8>().min_value(low + 1).max_value(n - 1));
    let shared = tc.draw(gs::integers::<u16>().min_value(0).max_value((1 << n) - 1))
        & !(1u16 << low)
        & !(1u16 << high);
    let upcard = tc.draw(gs::integers::<u8>().min_value(1).max_value(n));
    let remaining =
        tc.draw(gs::integers::<u16>().min_value(0).max_value((1 << n) - 1)) & !(1u16 << (upcard - 1));
    let state = GoofState {
        n,
        player_hand: shared | (1u16 << high),
        enemy_hand: shared | (1u16 << low),
        remaining,
        upcard,
        player_score: 0,
        enemy_score: 0,
    };
    let mut evaluator = ClosenessEvaluator { n };
    let evaluation = evaluator.evaluate(&state);
    assert!(
        evaluation.value > 0.0,
        "the stronger hand must be favored, got {}",
        evaluation.value
    );
    assert!(evaluation.value <= 1.0, "values stay in [-1, 1]");

    let mirrored = evaluator.evaluate(&mirror(&state));
    assert_eq!(mirrored.value, -evaluation.value, "value antisymmetry");
    assert_eq!(
        mirrored.player_priors, evaluation.enemy_priors,
        "priors swap"
    );
    assert_eq!(
        mirrored.enemy_priors, evaluation.player_priors,
        "priors swap"
    );
}

/// A fresh game is symmetric: identical hands, identical priors, and the
/// CRN chance seeds make the sampled root matrix exactly antisymmetric.
/// The cold equilibrium on it must therefore have value zero and equal
/// policies. Deterministic and exhaustive over every small fresh game.
///
/// The tolerances are looser than float noise because the RM+ dynamics
/// amplify it: the per-iteration value `p·M·p` is only zero up to
/// rounding, the two regret updates differ by twice that value, and
/// 2048 self-play iterations compound the asymmetry. The runs are
/// deterministic; observed drift peaks near 1e-7 in value and 1e-5 in
/// the policies (the value sits below the policy drift because the
/// near-equilibrium payoff rows are almost equalized).
#[test]
fn fresh_symmetric_games_solve_to_value_zero() {
    for n in 1..=6u8 {
        for first in 1..=n {
            let config = JointSearchConfig {
                expansion_budget: 1,
                minimum_expansion_budget: 1,
                ..JointSearchConfig::default()
            };
            let seed = 0xA11CE ^ (u64::from(n) << 8) ^ u64::from(first);
            let mut engine = SimultaneousTreeSearch::new(config.clone(), seed);
            let mut provider = GoofProvider;
            let mut evaluator = ClosenessEvaluator { n };
            let options = SearchOptions {
                sample_actions: false,
                router_score: 1.0,
            };
            let (result, tree) = engine.search_with_tree(
                &mut provider,
                &mut evaluator,
                GoofState::new(n, first),
                options,
            );
            let ctx = format!("fresh game n={n} first={first}");
            assert_joint_tree_invariants(&tree, &result, &config, &ctx);
            assert_eq!(
                result.transitions,
                u32::from(n) * u32::from(n),
                "{ctx}: root install covers the joint grid once"
            );
            assert!(
                result.root_value.abs() <= 1e-6,
                "{ctx}: symmetric game value {} should be zero",
                result.root_value
            );
            for action in 0..usize::from(n) {
                let drift = (result.player_policy[action] - result.enemy_policy[action]).abs();
                assert!(
                    drift <= 1e-4,
                    "{ctx}: symmetric policies diverge by {drift} at action {action}"
                );
            }
        }
    }
}
