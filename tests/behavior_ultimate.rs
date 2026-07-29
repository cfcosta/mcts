//! Outside-in behavioral guarantees on Ultimate Tic-Tac-Toe: a bigger
//! branching factor and much longer games than Tic-Tac-Toe, so this
//! exercises deep trees and long playouts.

mod support;

use mcts_rs::{Bump, Mcts, State};
use support::*;

/// Same exploration constant as the bundled example.
const C: f64 = 1.4142356237;

#[test]
fn search_always_returns_a_legal_action() {
    let positions = [
        ("start", uttt_after_plies(0)),
        ("midgame", uttt_after_plies(6)),
        ("late", uttt_after_plies(20)),
    ];
    for (name, position) in &positions {
        for n in [1, 50, 800] {
            let bump = Bump::new();
            let action = Mcts::new(&bump, position.clone(), C).search(n);
            assert!(
                position.get_legal_actions().contains(&action),
                "{name}, n={n}: illegal action {action:?}"
            );
        }
    }
}

#[test]
fn never_loses_to_a_random_opponent() {
    // Random play in Ultimate Tic-Tac-Toe is hopeless against even a small
    // search budget; at worst the game may be drawn. Calibration measured
    // 0 losses over 800 games at this budget, so a strict "never" is safe
    // for the 6 games played here.
    for seed in 0..3 {
        let as_x = play_game(
            UltimateTicTacToe::new(),
            mcts_policy(400, C),
            random_policy(seed),
        );
        assert_ne!(
            outcome(&as_x),
            Outcome::OWins,
            "seed {seed}: lost to a random opponent as X"
        );

        let as_o = play_game(
            UltimateTicTacToe::new(),
            random_policy(seed),
            mcts_policy(400, C),
        );
        assert_ne!(
            outcome(&as_o),
            Outcome::XWins,
            "seed {seed}: lost to a random opponent as O"
        );
    }
}

#[test]
fn self_play_terminates_with_valid_trees_at_every_move() {
    // A full self-play game at a small budget, re-checking every tree
    // invariant at every single move.
    let n = 150;
    let mut state = UltimateTicTacToe::new();
    let mut plies = 0usize;
    while !state.is_terminal() {
        let bump = Bump::new();
        let mut mcts = Mcts::new(&bump, state.clone(), C);
        let action = mcts.search(n);
        assert_tree_invariants(&mcts, n, &format!("self-play ply {plies}"));
        assert!(
            state.get_legal_actions().contains(&action),
            "ply {plies}: illegal action {action:?}"
        );
        state = state.step(action);
        plies += 1;
        assert!(plies <= 81, "game did not terminate within 81 plies");
    }
    // Any outcome is fine; the guarantee is legal play, valid trees, and
    // termination.
}
