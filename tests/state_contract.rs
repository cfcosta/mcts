//! Pins of the `State` trait's expansion contract: which methods the search
//! calls, and how the optional fast paths must relate to the required ones.

mod support;

use std::sync::atomic::{AtomicUsize, Ordering};

use mcts_rs::{Bump, Mcts, State};
use support::*;

#[test]
fn fill_legal_actions_default_appends_get_legal_actions() {
    let game = BanditGame::new();
    let mut buf = vec![Arm::Draw]; // sentinel: fill must append, never clear
    game.fill_legal_actions(&mut buf);
    assert_eq!(
        buf[0],
        Arm::Draw,
        "fill_legal_actions must not clear the buffer"
    );
    assert_eq!(
        buf[1..],
        game.get_legal_actions(),
        "default fill_legal_actions must append exactly get_legal_actions()"
    );
}

/// Counts calls to `get_legal_actions` while providing an allocation-free
/// `fill_legal_actions` override, so a test can observe which entry point
/// the search expands through. Three actions per state, draw after two plies.
#[derive(Debug, Clone)]
struct CountingGame {
    plies: u8,
}

static GET_LEGAL_ACTIONS_CALLS: AtomicUsize = AtomicUsize::new(0);

impl State for CountingGame {
    type Action = u8;

    fn default_action() -> Self::Action {
        0
    }

    fn player_has_won(&self, _player: usize) -> bool {
        false
    }

    fn is_terminal(&self) -> bool {
        self.plies == 2
    }

    fn get_legal_actions(&self) -> Vec<Self::Action> {
        GET_LEGAL_ACTIONS_CALLS.fetch_add(1, Ordering::Relaxed);
        if self.is_terminal() {
            Vec::new()
        } else {
            vec![0, 1, 2]
        }
    }

    fn fill_legal_actions(&self, actions: &mut Vec<Self::Action>) {
        if !self.is_terminal() {
            actions.extend([0, 1, 2]);
        }
    }

    // The rollout's default also falls back to get_legal_actions; override
    // it so the counter observes the expansion path in isolation.
    fn get_random_legal_action<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Self::Action {
        rng.gen_range(0..3)
    }

    fn to_play(&self) -> usize {
        usize::from(self.plies % 2)
    }

    fn step(&self, _action: Self::Action) -> Self {
        CountingGame {
            plies: self.plies + 1,
        }
    }

    fn reward(&self, _to_play: usize) -> f32 {
        0.0
    }

    fn render(&self) {}
}

#[test]
fn search_expands_through_fill_legal_actions_only() {
    GET_LEGAL_ACTIONS_CALLS.store(0, Ordering::Relaxed);
    let bump = Bump::new();
    let mut mcts = Mcts::new(&bump, CountingGame { plies: 0 }, 1.0);
    mcts.search(50);
    assert!(
        mcts.arena.len() > 4,
        "search must have expanded beyond the root"
    );
    assert_eq!(
        GET_LEGAL_ACTIONS_CALLS.load(Ordering::Relaxed),
        0,
        "the search must expand through fill_legal_actions, not get_legal_actions"
    );
}
