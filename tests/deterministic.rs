//! Exact pins of the search semantics on games designed so that the random
//! playouts cannot influence the asserted outcome. These tests have no
//! statistical component: any failure is a real behavior change, never a
//! flake.

mod support;

use mcts_rs::{Mcts, State};
use support::*;

#[test]
fn bandit_search_picks_the_winning_arm() {
    // Unvisited children have infinite UCB, so all three arms are visited
    // within the first three iterations. From then on their q values are
    // exactly +1 / 0 / -1, and the returned action (max q) is deterministic
    // for any n >= 3.
    for n in [3, 10, 500] {
        let action = Mcts::new(BanditGame::new(), 1.0).search(n);
        assert_eq!(action, Arm::Win, "n = {n}");
    }
}

#[test]
fn bandit_tree_has_exact_shape_and_values() {
    let n = 200;
    let mut mcts = Mcts::new(BanditGame::new(), 1.0);
    mcts.search(n);
    assert_tree_invariants(&mcts, n, "bandit n=200");

    // Root plus one child per arm; terminal children are never expanded.
    assert_eq!(mcts.arena.nodes.len(), 4);

    let root = mcts.arena.get_node(mcts.root_id);
    for &child_id in &root.children {
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
        let mut mcts = Mcts::new(ChainGame::new(length), 1.0);
        mcts.search(n);
        assert_tree_invariants(&mcts, n, &ctx);

        assert_eq!(mcts.arena.nodes.len(), 1 + length.min(n), "{ctx}");

        let mut depth = 0usize;
        let mut id = mcts.root_id;
        loop {
            let node = mcts.arena.get_node(id);
            let expected_visits = if depth == 0 { n } else { n - depth + 1 };
            assert_eq!(node.n, expected_visits, "{ctx}: visits at depth {depth}");
            assert_eq!(node.q, 0.0, "{ctx}: draws only, q must be exactly 0");
            match node.children.as_slice() {
                [] => break,
                [child] => {
                    id = *child;
                    depth += 1;
                }
                more => panic!("{ctx}: chain node has {} children", more.len()),
            }
        }
        assert_eq!(depth, length.min(n), "{ctx}: chain depth");
    }
}

#[test]
fn single_iteration_search_works_and_builds_a_minimal_tree() {
    let mut mcts = Mcts::new(ChainGame::new(8), 1.0);
    mcts.search(1);
    assert_tree_invariants(&mcts, 1, "chain n=1");
    // One iteration: the root is expanded once and one playout is recorded.
    assert_eq!(mcts.arena.nodes.len(), 2);
}

#[test]
#[should_panic(expected = "Option::unwrap()")]
fn search_with_zero_iterations_panics() {
    // Pins current behavior: with no iterations the root has no children,
    // so best-child selection unwraps None. If this contract is deliberately
    // changed (e.g. to return an error), update this test.
    Mcts::new(TicTacToe::new(), 0.5).search(0);
}

#[test]
#[should_panic(expected = "Option::unwrap()")]
fn search_from_a_terminal_state_panics() {
    // Same contract pin as above: callers must not ask for a move in a
    // finished game.
    let finished = ttt_after(&[(0, 0), (1, 1), (0, 1), (2, 2), (0, 2)]); // X wins the top row
    assert!(finished.is_terminal());
    assert_eq!(outcome(&finished), Outcome::XWins);
    Mcts::new(finished, 0.5).search(10);
}
