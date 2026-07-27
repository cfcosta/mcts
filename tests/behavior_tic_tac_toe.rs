//! Outside-in behavioral guarantees on Tic-Tac-Toe: whatever happens to the
//! internals, a search with a reasonable budget must keep playing correctly.
//!
//! The search uses unseeded random playouts, so these tests are statistical
//! in nature. Positions and iteration budgets are chosen so that the asserted
//! property holds with overwhelming probability (the tactical positions also
//! punish the wrong move immediately, which makes the value gap large). A
//! failure here should be treated as a real regression, not a flake.

mod support;

use mcts_rs::{Mcts, State};
use support::*;

/// Same exploration constant as the bundled example.
const C: f64 = 0.5;

#[test]
fn search_always_returns_a_legal_action() {
    // Includes a position with a single empty cell: the search must return
    // that cell for any iteration count.
    let almost_full = ttt_after(&[
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (2, 1),
        (1, 2),
        (1, 0),
        (2, 0),
    ]);
    assert_eq!(almost_full.get_legal_actions(), vec![(2, 2)]);

    let positions = [
        ("empty", ttt_after(&[])),
        ("one ply", ttt_after(&[(1, 1)])),
        ("midgame", ttt_after(&[(1, 1), (0, 0), (2, 0), (0, 2)])),
        ("almost full", almost_full),
    ];
    for (name, position) in &positions {
        for n in [1, 2, 7, 100] {
            let action = Mcts::new(position.clone(), C).search(n);
            assert!(
                position.get_legal_actions().contains(&action),
                "{name}, n={n}: illegal action {action:?}"
            );
        }
    }
}

#[test]
fn finds_the_immediate_winning_move() {
    // In every position it is X to play with a one-move win, while O also
    // threatens to win: missing the winning move loses rollouts quickly, so
    // the value gap is large.
    let cases = [
        ("row", ttt_after(&[(0, 0), (1, 0), (0, 1), (1, 1)]), (0, 2)),
        (
            "column",
            ttt_after(&[(0, 0), (0, 1), (1, 0), (1, 1)]),
            (2, 0),
        ),
        (
            "diagonal",
            ttt_after(&[(0, 0), (0, 2), (1, 1), (1, 2)]),
            (2, 2),
        ),
    ];
    for (name, position, winning_move) in cases {
        let action = Mcts::new(position, C).search(2000);
        assert_eq!(action, winning_move, "{name}: must play the winning move");
    }
}

#[test]
fn blocks_the_opponents_winning_move() {
    // X: (1,1), (2,2); O: (0,0), (0,1). X has no immediate win and O
    // threatens (0,2); any other X move lets O win on the spot.
    let position = ttt_after(&[(1, 1), (0, 0), (2, 2), (0, 1)]);
    for attempt in 0..3 {
        let action = Mcts::new(position.clone(), C).search(5000);
        assert_eq!(action, (0, 2), "attempt {attempt}: must block O's win");
    }
}

#[test]
fn prefers_winning_to_blocking() {
    // X can win at (0,2); O threatens (2,2). Taking the win ends the game,
    // so it must be preferred over blocking.
    let position = ttt_after(&[(0, 0), (2, 0), (0, 1), (2, 1)]);
    let action = Mcts::new(position, C).search(2000);
    assert_eq!(action, (0, 2), "must take the win, not block");
}

#[test]
fn self_play_is_almost_always_a_draw() {
    // Tic-Tac-Toe is a draw under correct play. The current implementation
    // picks the final move by max q (not max visits), so a rare blunder is
    // inherent: calibration over 400 games measured a ~0.5% decisive rate at
    // this budget (and it does not vanish at higher budgets). Allowing 2
    // decisive games out of 12 makes the false-failure rate ~1e-4 while any
    // real play-quality regression (decisive rates of 10%+) still fails with
    // near certainty.
    let games = 12;
    let decisive = (0..games)
        .filter(|_| {
            let final_state = play_game(TicTacToe::new(), mcts_policy(2000, C), mcts_policy(2000, C));
            outcome(&final_state) != Outcome::Draw
        })
        .count();
    assert!(
        decisive <= 2,
        "{decisive}/{games} self-play games were decisive; expected almost all draws"
    );
}

#[test]
fn practically_never_loses_to_a_random_opponent() {
    // As either player, the search must essentially never lose to uniformly
    // random play (the opponent is seeded, so its moves are reproducible).
    // Calibration over 800 games measured a ~0.1% loss rate as O; allowing a
    // single loss across 24 games keeps the false-failure rate ~4e-4.
    let mut losses = 0;
    for seed in 0..12 {
        let as_x = play_game(TicTacToe::new(), mcts_policy(1500, C), random_policy(seed));
        if outcome(&as_x) == Outcome::OWins {
            eprintln!("seed {seed}: lost to a random opponent as X");
            losses += 1;
        }

        let as_o = play_game(TicTacToe::new(), random_policy(seed), mcts_policy(1500, C));
        if outcome(&as_o) == Outcome::XWins {
            eprintln!("seed {seed}: lost to a random opponent as O");
            losses += 1;
        }
    }
    assert!(
        losses <= 1,
        "lost {losses}/24 games to a random opponent; expected at most 1"
    );
}
