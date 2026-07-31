//! Exact pins of the search semantics on games designed so that the random
//! playouts cannot influence the asserted outcome. These tests have no
//! statistical component: any failure is a real behavior change, never a
//! flake.

mod support;

use mcts_rs::{Bump, Mcts, State};
use support::*;

#[test]
fn bandit_search_picks_the_winning_arm() {
    // Unvisited children have infinite UCB, so all three arms are visited
    // within the first three iterations. From then on their q values are
    // exactly +1 / 0 / -1, and the returned action (max q) is deterministic
    // for any n >= 3.
    for n in [3, 10, 500] {
        let bump = Bump::new();
        let action = Mcts::new(&bump, BanditGame::new(), 1.0).search(n);
        assert_eq!(action, Arm::Win, "n = {n}");
    }
}

#[test]
fn bandit_tree_has_exact_shape_and_values() {
    let n = 200;
    let bump = Bump::new();
    let mut mcts = Mcts::new(&bump, BanditGame::new(), 1.0);
    mcts.search(n);
    assert_tree_invariants(&mcts, n, "bandit n=200");

    // Root plus one child per arm; terminal children are never expanded.
    assert_eq!(mcts.arena.len(), 4);

    let root = mcts.arena.get_node(mcts.root_id);
    for child_id in root.children.ids() {
        let child = mcts.arena.get_node(child_id);
        let expected_q = match child.action {
            Arm::Win => 1.0,
            Arm::Draw => 0.0,
            Arm::Lose => -1.0,
        };
        // Every playout through an arm yields the same reward, so the mean
        // is exact, with no floating-point slack needed.
        assert_eq!(child.q, expected_q, "arm {:?}", child.action);
        assert!(child.n >= 1, "arm {:?} was never explored", child.action);
    }
}

#[test]
fn chain_tree_grows_one_node_per_iteration_until_terminal() {
    // With a single legal action everywhere, selection always walks to the
    // deepest node, expansion adds exactly one child, and backpropagation
    // updates the whole path. The resulting tree is a chain with fully
    // determined visit counts: the node at depth d is created on iteration d
    // and visited on every iteration from then on.
    for (length, n) in [(64, 50), (3, 100)] {
        let ctx = format!("chain length={length} n={n}");
        let bump = Bump::new();
        let mut mcts = Mcts::new(&bump, ChainGame::new(length), 1.0);
        mcts.search(n);
        assert_tree_invariants(&mcts, n, &ctx);

        assert_eq!(mcts.arena.len(), 1 + length.min(n), "{ctx}");

        let mut depth = 0usize;
        let mut id = mcts.root_id;
        loop {
            let node = mcts.arena.get_node(id);
            let expected_visits = if depth == 0 { n } else { n - depth + 1 };
            assert_eq!(node.n, expected_visits, "{ctx}: visits at depth {depth}");
            assert_eq!(node.q, 0.0, "{ctx}: draws only, q must be exactly 0");
            match node.children.len() {
                0 => break,
                1 => {
                    id = node.children.ids().next().unwrap();
                    depth += 1;
                }
                more => panic!("{ctx}: chain node has {more} children"),
            }
        }
        assert_eq!(depth, length.min(n), "{ctx}: chain depth");
    }
}

#[test]
fn single_iteration_search_works_and_builds_a_minimal_tree() {
    let bump = Bump::new();
    let mut mcts = Mcts::new(&bump, ChainGame::new(8), 1.0);
    mcts.search(1);
    assert_tree_invariants(&mcts, 1, "chain n=1");
    // One iteration: the root is expanded once and one playout is recorded.
    assert_eq!(mcts.arena.len(), 2);
}

/// Every arm loses for the mover, so any visited root child ends at q = -1
/// while unvisited arms stay at q = 0. Which arm the first iteration visits
/// depends on the RNG, but the asserted property does not.
#[derive(Debug, Clone)]
struct LosingGame {
    taken: Option<u8>,
}

impl State for LosingGame {
    type Action = u8;

    fn default_action() -> Self::Action {
        0
    }

    fn player_has_won(&self, player: usize) -> bool {
        self.taken.is_some() && player == 1
    }

    fn is_terminal(&self) -> bool {
        self.taken.is_some()
    }

    fn get_legal_actions(&self) -> Vec<Self::Action> {
        if self.taken.is_some() {
            Vec::new()
        } else {
            vec![0, 1, 2]
        }
    }

    fn to_play(&self) -> usize {
        usize::from(self.taken.is_some())
    }

    fn step(&self, action: Self::Action) -> Self {
        LosingGame {
            taken: Some(action),
        }
    }

    fn reward(&self, to_play: usize) -> f32 {
        if self.player_has_won(to_play) {
            -1.0
        } else if self.player_has_won(1 - to_play) {
            1.0
        } else {
            0.0
        }
    }

    fn render(&self) {}
}

#[test]
fn unvisited_root_children_count_as_q_zero_in_the_final_pick() {
    // A single iteration visits exactly one arm, which ends at q = -1; the
    // other two arms are never visited. The final pick must value those
    // unvisited children at exactly q = 0 — never NaN or -inf — and so
    // return one of them over the visited losing arm, for any RNG outcome.
    for round in 0..32 {
        let bump = Bump::new();
        let mut mcts = Mcts::new(&bump, LosingGame { taken: None }, 1.0);
        let action = mcts.search(1);
        let root = mcts.arena.get_node(mcts.root_id);
        let chosen = root
            .children
            .ids()
            .find(|&id| mcts.arena.get_node(id).action == action)
            .unwrap();
        assert_eq!(
            mcts.arena.get_node(chosen).n,
            0,
            "round {round}: the final pick must prefer an unvisited arm (q = 0) \
             over the visited losing arm (q = -1)"
        );
    }
}

#[test]
fn losing_game_means_stay_exact_for_any_iteration_count() {
    // Every playout returns the same reward, so each node's mean is exact
    // at every visit count: a pin on the mean arithmetic and the per-level
    // sign flip that must survive any change to how statistics are stored.
    for n in [1, 7, 64, 1000] {
        let bump = Bump::new();
        let mut mcts = Mcts::new(&bump, LosingGame { taken: None }, 1.0);
        mcts.search(n);
        let root = mcts.arena.get_node(mcts.root_id);
        assert_eq!(root.q, 1.0, "n={n}: root mean must be exactly +1");
        for id in root.children.ids() {
            let child = mcts.arena.get_node(id);
            if child.n > 0 {
                assert_eq!(child.q, -1.0, "n={n}: visited arm mean must be exactly -1");
            } else {
                assert_eq!(child.q, 0.0, "n={n}: unvisited arm must stay at 0");
            }
        }
    }
}

#[test]
#[should_panic(expected = "Option::unwrap()")]
fn search_with_zero_iterations_panics() {
    // Pins current behavior: with no iterations the root has no children,
    // so best-child selection unwraps None. If this contract is deliberately
    // changed (e.g. to return an error), update this test.
    let bump = Bump::new();
    Mcts::new(&bump, TicTacToe::new(), 0.5).search(0);
}

#[test]
#[should_panic(expected = "Option::unwrap()")]
fn search_from_a_terminal_state_panics() {
    // Same contract pin as above: callers must not ask for a move in a
    // finished game.
    let finished = ttt_after(&[(0, 0), (1, 1), (0, 1), (2, 2), (0, 2)]); // X wins the top row
    assert!(finished.is_terminal());
    assert_eq!(outcome(&finished), Outcome::XWins);
    let bump = Bump::new();
    Mcts::new(&bump, finished, 0.5).search(10);
}
