//! Structural invariants of the search tree, checked over a grid of games,
//! positions, exploration constants, and iteration counts. See
//! `support::assert_tree_invariants` for the exact list of properties.

mod support;

use mcts_rs::Mcts;
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
                let mut mcts = Mcts::new(position.clone(), c);
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
            let mut mcts = Mcts::new(position.clone(), 1.4);
            mcts.search(n);
            assert_tree_invariants(&mcts, n, &format!("uttt {name}, n={n}"));
        }
    }
}

#[test]
fn invariants_hold_for_synthetic_games() {
    for n in [1, 5, 100] {
        let mut mcts = Mcts::new(ChainGame::new(16), 1.0);
        mcts.search(n);
        assert_tree_invariants(&mcts, n, &format!("chain(16), n={n}"));
    }
    for n in [1, 3, 50] {
        let mut mcts = Mcts::new(BanditGame::new(), 1.0);
        mcts.search(n);
        assert_tree_invariants(&mcts, n, &format!("bandit, n={n}"));
    }
    for n in [1, 64, 200] {
        let mut mcts = Mcts::new(WideGame::new(64), 1.0);
        mcts.search(n);
        assert_tree_invariants(&mcts, n, &format!("wide(64), n={n}"));
    }
}
