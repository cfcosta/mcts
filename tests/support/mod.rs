//! Shared helpers for the integration test and benchmark suites.
//!
//! This module is compiled into every test binary (via `mod support;`) and
//! into the benchmark binary (via `#[path]`), so everything here is written
//! against the public API of the crate only.
#![allow(dead_code)]

use std::collections::VecDeque;
use std::fmt::Debug;

use mcts_rs::{Bump, Mcts, State};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

pub mod games;
pub mod joint;

pub use games::{TicTacToe, UltimateTicTacToe};

/// A game with exactly one legal action per state, ending in a draw after
/// `length` moves. Nothing about it is random, so the search tree it produces
/// is fully determined and tests can assert exact shapes and visit counts.
/// It also isolates the cost of selection/expansion/backpropagation for
/// benchmarks, since the state itself is nearly free.
#[derive(Debug, Clone)]
pub struct ChainGame {
    pub length: usize,
    pub remaining: usize,
}

impl ChainGame {
    pub fn new(length: usize) -> Self {
        ChainGame {
            length,
            remaining: length,
        }
    }
}

impl State for ChainGame {
    type Action = ();

    fn default_action() -> Self::Action {}

    fn player_has_won(&self, _player: usize) -> bool {
        false
    }

    fn is_terminal(&self) -> bool {
        self.remaining == 0
    }

    fn get_legal_actions(&self) -> Vec<Self::Action> {
        if self.remaining == 0 {
            Vec::new()
        } else {
            vec![()]
        }
    }

    fn fill_legal_actions(&self, actions: &mut Vec<Self::Action>) {
        if self.remaining > 0 {
            actions.push(());
        }
    }

    fn to_play(&self) -> usize {
        (self.length - self.remaining) % 2
    }

    fn step(&self, _action: Self::Action) -> Self {
        assert!(self.remaining > 0, "step on a terminal ChainGame state");
        ChainGame {
            length: self.length,
            remaining: self.remaining - 1,
        }
    }

    fn reward(&self, _to_play: usize) -> f32 {
        0.0
    }

    fn render(&self) {}
}

/// A single decision among three arms with fixed outcomes: player 0 moves
/// once and the game ends. Because every playout through an arm yields the
/// same reward, the q values of the root's children are exact (+1 / 0 / -1)
/// as soon as each arm has been visited once, which makes the best action
/// fully deterministic despite the random playouts.
///
/// The winning arm is deliberately in the middle of the legal-action list so
/// that neither a "first child" nor a "last child" implementation passes by
/// accident.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    Lose,
    Win,
    Draw,
}

#[derive(Debug, Clone)]
pub struct BanditGame {
    pub taken: Option<Arm>,
}

impl BanditGame {
    pub fn new() -> Self {
        BanditGame { taken: None }
    }
}

impl State for BanditGame {
    type Action = Arm;

    fn default_action() -> Self::Action {
        Arm::Draw
    }

    fn player_has_won(&self, player: usize) -> bool {
        match self.taken {
            Some(Arm::Win) => player == 0,
            Some(Arm::Lose) => player == 1,
            _ => false,
        }
    }

    fn is_terminal(&self) -> bool {
        self.taken.is_some()
    }

    fn get_legal_actions(&self) -> Vec<Self::Action> {
        if self.taken.is_some() {
            Vec::new()
        } else {
            vec![Arm::Lose, Arm::Win, Arm::Draw]
        }
    }

    fn to_play(&self) -> usize {
        usize::from(self.taken.is_some())
    }

    fn step(&self, action: Self::Action) -> Self {
        BanditGame {
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

/// One decision among `width` arms, all draws, terminal after one move.
/// Exercises wide expansion and UCB selection across many siblings.
#[derive(Debug, Clone)]
pub struct WideGame {
    pub width: usize,
    pub taken: Option<usize>,
}

impl WideGame {
    pub fn new(width: usize) -> Self {
        WideGame { width, taken: None }
    }
}

impl State for WideGame {
    type Action = usize;

    fn default_action() -> Self::Action {
        0
    }

    fn player_has_won(&self, _player: usize) -> bool {
        false
    }

    fn is_terminal(&self) -> bool {
        self.taken.is_some()
    }

    fn get_legal_actions(&self) -> Vec<Self::Action> {
        if self.taken.is_some() {
            Vec::new()
        } else {
            (0..self.width).collect()
        }
    }

    fn fill_legal_actions(&self, actions: &mut Vec<Self::Action>) {
        if self.taken.is_none() {
            actions.extend(0..self.width);
        }
    }

    fn to_play(&self) -> usize {
        usize::from(self.taken.is_some())
    }

    fn step(&self, action: Self::Action) -> Self {
        WideGame {
            width: self.width,
            taken: Some(action),
        }
    }

    fn reward(&self, _to_play: usize) -> f32 {
        0.0
    }

    fn render(&self) {}
}

/// The result of a finished two-player game, from the fixed player numbering
/// (player 0 = X, player 1 = O).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    XWins,
    OWins,
    Draw,
}

pub fn outcome<S: State>(state: &S) -> Outcome {
    if state.player_has_won(0) {
        Outcome::XWins
    } else if state.player_has_won(1) {
        Outcome::OWins
    } else {
        Outcome::Draw
    }
}

/// Builds a Tic-Tac-Toe position by applying `moves` (row, col) from the
/// empty board, alternating players starting with X. Panics if a setup move
/// is illegal or the game ends before all moves are applied, so tests cannot
/// silently run on a different position than intended.
pub fn ttt_after(moves: &[(u8, u8)]) -> TicTacToe {
    let mut state = TicTacToe::new();
    for &m in moves {
        assert!(
            !state.is_terminal(),
            "setup position ended early before move {m:?}"
        );
        assert!(
            state.get_legal_actions().contains(&m),
            "illegal setup move {m:?}"
        );
        state = state.step(m);
    }
    state
}

/// Builds an Ultimate Tic-Tac-Toe position by playing `plies` moves from the
/// start, always choosing the first legal action. Deterministic and always
/// legal by construction.
pub fn uttt_after_plies(plies: usize) -> UltimateTicTacToe {
    let mut state = UltimateTicTacToe::new();
    for _ in 0..plies {
        assert!(!state.is_terminal(), "setup position ended early");
        let action = state.get_legal_actions()[0];
        state = state.step(action);
    }
    state
}

/// Plays a full game between two policies, asserting at every ply that the
/// chosen move is legal in the current state.
pub fn play_game<S, F, G>(start: S, mut player_x: F, mut player_o: G) -> S
where
    S: State + Clone,
    S::Action: PartialEq + Debug,
    F: FnMut(&S) -> S::Action,
    G: FnMut(&S) -> S::Action,
{
    let mut state = start;
    let mut plies = 0usize;
    while !state.is_terminal() {
        let action = if state.to_play() == 0 {
            player_x(&state)
        } else {
            player_o(&state)
        };
        assert!(
            state.get_legal_actions().contains(&action),
            "policy for player {} chose illegal action {action:?}",
            state.to_play()
        );
        state = state.step(action);
        plies += 1;
        assert!(plies <= 1000, "game did not terminate within 1000 plies");
    }
    state
}

/// A policy that runs a fresh MCTS search for every move, reusing one bump
/// arena across moves the way a real caller would.
pub fn mcts_policy<S>(iterations: usize, c: f64) -> impl FnMut(&S) -> S::Action
where
    S: State + Clone + Debug,
{
    let mut bump = Bump::new();
    move |state| {
        let action = Mcts::new(&bump, state.clone(), c).search(iterations);
        bump.reset();
        action
    }
}

/// A uniformly random policy with a fixed seed, so the opponent's moves are
/// reproducible across runs.
pub fn random_policy<S: State>(seed: u64) -> impl FnMut(&S) -> S::Action {
    let mut rng = StdRng::seed_from_u64(seed);
    move |state| {
        *state
            .get_legal_actions()
            .choose(&mut rng)
            .expect("no legal actions in a non-terminal state")
    }
}

/// Checks every structural invariant that `Mcts::search(iterations)` must
/// leave the tree in, for any game and any iteration count:
///
/// - the root has no parent and exactly `iterations` visits;
/// - parent/child links are mutually consistent and the arena contains
///   exactly the nodes reachable from the root (no orphans, no sharing);
/// - `q` is bounded by the game's reward range ([-1, 1] for all games used
///   in the tests; the exact mean semantics are pinned in
///   `tests/deterministic.rs`);
/// - unvisited nodes are untouched leaves;
/// - expansion is all-or-nothing: an expanded node has exactly one child per
///   legal action, and terminal nodes are never expanded;
/// - every iteration passes through exactly one child of the root.
///
/// `ctx` is prefixed to every failure message to identify the scenario.
pub fn assert_tree_invariants<S>(mcts: &Mcts<S>, iterations: usize, ctx: &str)
where
    S: State,
    S::Action: PartialEq + Debug,
{
    let node_count = mcts.arena.len();
    let root_id = mcts.root_id;
    let root = mcts.arena.get_node(root_id);

    assert!(root.parent.is_none(), "{ctx}: root must have no parent");
    assert_eq!(
        root.n, iterations,
        "{ctx}: root visits must equal search iterations"
    );

    let mut seen = vec![false; node_count];
    seen[root_id] = true;
    let mut queue = VecDeque::from([root_id]);
    let mut reachable = 0usize;
    while let Some(id) = queue.pop_front() {
        reachable += 1;
        for child_id in mcts.arena.get_node(id).children.ids() {
            let child = mcts.arena.get_node(child_id);
            assert_eq!(
                child.parent,
                Some(id),
                "{ctx}: node {child_id} does not point back to its parent {id}"
            );
            assert!(
                !seen[child_id],
                "{ctx}: node {child_id} appears in two children lists"
            );
            seen[child_id] = true;
            queue.push_back(child_id);
        }
    }
    assert_eq!(
        reachable, node_count,
        "{ctx}: arena contains nodes unreachable from the root"
    );

    for id in 0..node_count {
        let node = mcts.arena.get_node(id);
        if node.n == 0 {
            assert_eq!(node.q, 0.0, "{ctx}: unvisited node {id} has a nonzero q");
            assert!(
                node.children.is_empty(),
                "{ctx}: unvisited node {id} must not have been expanded"
            );
        }
        assert!(
            node.q.abs() <= 1.0 + 1e-9,
            "{ctx}: node {id} has q = {} outside the reward range",
            node.q
        );

        if node.state.is_terminal() {
            assert!(
                node.children.is_empty(),
                "{ctx}: terminal node {id} must never be expanded"
            );
        }

        if !node.children.is_empty() {
            let legal = node.state.get_legal_actions();
            assert_eq!(
                node.children.len(),
                legal.len(),
                "{ctx}: expanded node {id} must have one child per legal action"
            );
            assert!(node.n >= 1, "{ctx}: expanded node {id} was never visited");
            for &action in &legal {
                let matching = node
                    .children
                    .ids()
                    .filter(|&c| mcts.arena.get_node(c).action == action)
                    .count();
                assert_eq!(
                    matching, 1,
                    "{ctx}: expanded node {id} must have exactly one child for action {action:?}"
                );
            }
        }
    }

    if iterations > 0 && !root.state.is_terminal() {
        let child_visits: usize = root.children.ids().map(|c| mcts.arena.get_node(c).n).sum();
        assert_eq!(
            child_visits, iterations,
            "{ctx}: every iteration must pass through exactly one child of the root"
        );
    }
}
