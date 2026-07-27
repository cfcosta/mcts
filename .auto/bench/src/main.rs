use std::hint::black_box;
use std::time::Instant;

use mcts_rs::games::{TicTacToe, UltimateTicTacToe};
use mcts_rs::{Mcts, State};
use rand::seq::SliceRandom;

const TTT_C: f64 = 0.5;
const UTTT_C: f64 = 1.4142356237;

#[derive(Debug, Clone)]
struct ChainGame {
    length: usize,
    remaining: usize,
}

impl ChainGame {
    fn new(length: usize) -> Self {
        Self {
            length,
            remaining: length,
        }
    }
}

impl State for ChainGame {
    type Action = ();

    fn default_action() -> Self::Action {}
    fn player_has_won(&self, _: usize) -> bool {
        false
    }
    fn is_terminal(&self) -> bool {
        self.remaining == 0
    }
    fn get_legal_actions(&self) -> Vec<Self::Action> {
        if self.is_terminal() {
            Vec::new()
        } else {
            vec![()]
        }
    }
    fn to_play(&self) -> usize {
        (self.length - self.remaining) % 2
    }
    fn step(&self, _: Self::Action) -> Self {
        Self {
            length: self.length,
            remaining: self.remaining - 1,
        }
    }
    fn reward(&self, _: usize) -> f32 {
        0.0
    }
    fn render(&self) {}
}

#[derive(Debug, Clone)]
struct WideGame {
    width: usize,
    taken: Option<usize>,
}

impl WideGame {
    fn new(width: usize) -> Self {
        Self { width, taken: None }
    }
}

impl State for WideGame {
    type Action = usize;

    fn default_action() -> Self::Action {
        0
    }
    fn player_has_won(&self, _: usize) -> bool {
        false
    }
    fn is_terminal(&self) -> bool {
        self.taken.is_some()
    }
    fn get_legal_actions(&self) -> Vec<Self::Action> {
        if self.is_terminal() {
            Vec::new()
        } else {
            (0..self.width).collect()
        }
    }
    fn to_play(&self) -> usize {
        usize::from(self.taken.is_some())
    }
    fn step(&self, action: Self::Action) -> Self {
        Self {
            width: self.width,
            taken: Some(action),
        }
    }
    fn reward(&self, _: usize) -> f32 {
        0.0
    }
    fn render(&self) {}
}

fn ttt_midgame() -> TicTacToe {
    let mut state = TicTacToe::new();
    for action in [(1, 1), (0, 0), (2, 0), (0, 2)] {
        state = state.step(action);
    }
    state
}

fn uttt_midgame() -> UltimateTicTacToe {
    let mut state = UltimateTicTacToe::new();
    for _ in 0..6 {
        let action = state.get_legal_actions()[0];
        state = state.step(action);
    }
    state
}

fn random_playout<S: State + Clone>(start: &S) -> f32 {
    let mut state = start.clone();
    while !state.is_terminal() {
        let actions = state.get_legal_actions();
        let action = *actions.choose(&mut rand::thread_rng()).unwrap();
        state = state.step(action);
    }
    state.reward(0)
}

fn median_ns<F, T>(batch: usize, rounds: usize, mut f: F) -> f64
where
    F: FnMut() -> T,
{
    for _ in 0..(batch / 10).max(1) {
        black_box(f());
    }

    let mut samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        for _ in 0..batch {
            black_box(f());
        }
        samples.push(start.elapsed().as_secs_f64() * 1e9 / batch as f64);
    }
    samples.sort_unstable_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn main() {
    let ttt_empty = TicTacToe::new();
    let ttt_mid = ttt_midgame();
    let uttt_empty = UltimateTicTacToe::new();
    let uttt_mid = uttt_midgame();
    let uttt_step_action = uttt_mid.get_legal_actions()[0];

    let metrics = [
        (
            "ttt_empty_ns",
            median_ns(24, 5, || {
                Mcts::new(black_box(ttt_empty.clone()), TTT_C).search(10_000)
            }),
        ),
        (
            "ttt_mid_ns",
            median_ns(64, 5, || {
                Mcts::new(black_box(ttt_mid.clone()), TTT_C).search(10_000)
            }),
        ),
        (
            "uttt_empty_ns",
            median_ns(20, 5, || {
                Mcts::new(black_box(uttt_empty.clone()), UTTT_C).search(1_000)
            }),
        ),
        (
            "uttt_mid_ns",
            median_ns(20, 5, || {
                Mcts::new(black_box(uttt_mid.clone()), UTTT_C).search(1_000)
            }),
        ),
        (
            "deep_chain_ns",
            median_ns(24, 5, || Mcts::new(ChainGame::new(512), 1.0).search(2_000)),
        ),
        (
            "wide_512_ns",
            median_ns(32, 5, || Mcts::new(WideGame::new(512), 1.0).search(1_024)),
        ),
        (
            "uttt_rollout_ns",
            median_ns(20_000, 5, || random_playout(black_box(&uttt_empty))),
        ),
        (
            "uttt_step_ns",
            median_ns(2_000_000, 5, || {
                black_box(&uttt_mid).step(black_box(uttt_step_action))
            }),
        ),
    ];

    for (name, value) in metrics {
        println!("METRIC {name}={value:.3}");
    }
    let geomean = (metrics
        .iter()
        .map(|(_, value)| value.ln())
        .sum::<f64>()
        / metrics.len() as f64)
        .exp();
    println!("METRIC geomean_ns={geomean:.3}");
}
