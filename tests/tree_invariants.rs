//! Structural invariants of the search tree, checked over a grid of games,
//! positions, exploration constants, and iteration counts. See
//! `support::assert_tree_invariants` for the exact list of properties.

mod support;

use mcts_rs::{Bump, Mcts};
use support::*;

#[test]
fn invariants_hold_for_tic_tac_toe_positions() {
    let positions = [
        ("empty", ttt_after(&[])),
        ("one ply", ttt_after(&[(1, 1)])),
        ("midgame", ttt_after(&[(1, 1), (0, 0), (2, 0), (0, 2)])),
        (
            "late game",
            ttt_after(&[(1, 1), (0, 0), (2, 0), (0, 2), (0, 1), (2, 1)]),
        ),
    ];
    for (name, position) in &positions {
        for n in [1, 2, 3, 10, 137, 2000] {
            for c in [0.0, 0.5, 2.0] {
                let bump = Bump::new();
                let mut mcts = Mcts::new(&bump, position.clone(), c);
                mcts.search(n);
                assert_tree_invariants(&mcts, n, &format!("ttt {name}, n={n}, c={c}"));
            }
        }
    }
}

#[test]
fn invariants_hold_for_ultimate_tic_tac_toe_positions() {
    let positions = [
        ("start", uttt_after_plies(0)),
        ("midgame", uttt_after_plies(6)),
    ];
    for (name, position) in &positions {
        for n in [1, 10, 400] {
            let bump = Bump::new();
            let mut mcts = Mcts::new(&bump, position.clone(), 1.4);
            mcts.search(n);
            assert_tree_invariants(&mcts, n, &format!("uttt {name}, n={n}"));
        }
    }
}

#[test]
fn invariants_hold_across_repeated_searches_on_one_tree() {
    // The tree is reusable: a second search call continues where the first
    // stopped, and every invariant must hold for the accumulated totals.
    let bump = Bump::new();
    let mut mcts = Mcts::new(&bump, TicTacToe::new(), 0.5);
    for total in [50, 100, 150] {
        mcts.search(50);
        assert_tree_invariants(&mcts, total, &format!("ttt accumulated n={total}"));
    }
}

#[test]
fn invariants_hold_for_synthetic_games() {
    for n in [1, 5, 100] {
        let bump = Bump::new();
        let mut mcts = Mcts::new(&bump, ChainGame::new(16), 1.0);
        mcts.search(n);
        assert_tree_invariants(&mcts, n, &format!("chain(16), n={n}"));
    }
    for n in [1, 3, 50] {
        let bump = Bump::new();
        let mut mcts = Mcts::new(&bump, BanditGame::new(), 1.0);
        mcts.search(n);
        assert_tree_invariants(&mcts, n, &format!("bandit, n={n}"));
    }
    for n in [1, 64, 200] {
        let bump = Bump::new();
        let mut mcts = Mcts::new(&bump, WideGame::new(64), 1.0);
        mcts.search(n);
        assert_tree_invariants(&mcts, n, &format!("wide(64), n={n}"));
    }
}
