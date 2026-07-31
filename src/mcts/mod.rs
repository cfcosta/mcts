pub mod arena;
pub mod node;

use crate::state::State;
use arena::Arena;
use node::{Children, Node, NO_PARENT};

use bumpalo::Bump;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

fn shared_lookup_table(
    tables: &'static OnceLock<Mutex<HashMap<usize, &'static [f64]>>>,
    required: usize,
    value: fn(usize) -> f64,
) -> &'static [f64] {
    let len = required
        .saturating_add(1)
        .checked_next_power_of_two()
        .unwrap();
    let tables = tables.get_or_init(|| Mutex::new(HashMap::new()));
    let mut tables = tables.lock().unwrap();
    *tables.entry(len).or_insert_with(|| {
        let values = (0..len).map(value).collect::<Vec<_>>().into_boxed_slice();
        Box::leak(values)
    })
}

fn inverse_sqrt_table(required: usize) -> &'static [f64] {
    static TABLES: OnceLock<Mutex<HashMap<usize, &'static [f64]>>> = OnceLock::new();
    shared_lookup_table(&TABLES, required, |n| {
        if n == 0 {
            f64::INFINITY
        } else {
            1.0 / (n as f64).sqrt()
        }
    })
}

fn sqrt_log_table(required: usize) -> &'static [f64] {
    static TABLES: OnceLock<Mutex<HashMap<usize, &'static [f64]>>> = OnceLock::new();
    shared_lookup_table(&TABLES, required, |n| {
        if n == 0 {
            0.0
        } else {
            (n as f64).ln().sqrt()
        }
    })
}

pub struct Mcts<'b, S: State> {
    pub arena: Arena<'b, S>,
    pub root_id: usize,
    c: f64,
    inverse_sqrt: &'static [f64],
    sqrt_log: &'static [f64],
}

impl<'b, S: State + std::fmt::Debug + std::clone::Clone> Mcts<'b, S> {
    /// Creates a search tree rooted at `state`, allocating nodes in `bump`.
    ///
    /// Callers running repeated searches should keep one [`Bump`] alive and
    /// [`reset`](Bump::reset) it between searches, once the `Mcts` is
    /// dropped: the arena then reuses its retained chunk and steady-state
    /// searches never touch the system allocator.
    pub fn new(bump: &'b Bump, state: S, c: f64) -> Self {
        let mut arena: Arena<'b, S> = Arena::new(bump);
        let root_id: usize = arena.push(state, S::default_action(), None);
        Mcts {
            arena,
            root_id,
            c,
            inverse_sqrt: &[f64::INFINITY],
            sqrt_log: &[0.0],
        }
    }

    pub fn search(&mut self, n: usize) -> S::Action {
        let current_visits = self.arena.nodes[self.root_id].hot.n as usize;
        let cached_visits = current_visits.saturating_add(n);
        let mut inverse_ready = self.inverse_sqrt.len() > cached_visits;
        let mut factors_ready = self.sqrt_log.len() > cached_visits;
        // A search-local generator: unlike `thread_rng`, every draw avoids
        // the thread-local access, reseeding checks, and ChaCha block
        // machinery — the rollout loop only needs statistical uniformity,
        // not cryptographic quality. Seeding from `thread_rng` (rather than
        // the OS) keeps search setup free of syscalls.
        let mut rng =
            SmallRng::from_rng(rand::thread_rng()).expect("thread_rng cannot fail to produce a seed");
        // Reused across every expansion of this search; cleared before each
        // fill, so after the first few expansions it never reallocates.
        let mut legal_buf: Vec<S::Action> = Vec::new();

        for _ in 0..n {
            let mut selected_id: usize = self.select();
            if !self.arena.nodes[selected_id].state.is_terminal() {
                self.expand(selected_id, &mut legal_buf);
                let children = self.arena.nodes[selected_id].hot.children.ids();
                if children.len() > 1 {
                    if !inverse_ready {
                        self.extend_inverse_sqrt(cached_visits);
                        inverse_ready = true;
                    }
                    // Parent factors are most useful when a long search or a
                    // very wide scan amortizes their eager construction.
                    if !factors_ready && (cached_visits >= 4_096 || children.len() >= 256) {
                        self.extend_sqrt_log(cached_visits);
                        factors_ready = true;
                    }
                }
                selected_id = children.start + rng.gen_range(0..children.len());
            }
            let reward: f64 = self.simulate(selected_id, &mut rng);
            self.backprop(selected_id, reward);
        }
        let best_child: usize = self.arena.nodes[self.root_id]
            .hot
            .children
            .ids()
            .max_by(|&a, &b| {
                let node_a_score = self.arena.nodes[a].hot.q;
                let node_b_score = self.arena.nodes[b].hot.q;
                node_a_score.partial_cmp(&node_b_score).unwrap()
            })
            .unwrap();

        self.arena.nodes[best_child].action
    }

    #[cold]
    #[inline(never)]
    fn extend_inverse_sqrt(&mut self, cached_visits: usize) {
        self.inverse_sqrt = inverse_sqrt_table(cached_visits);
    }

    #[cold]
    #[inline(never)]
    fn extend_sqrt_log(&mut self, cached_visits: usize) {
        self.sqrt_log = sqrt_log_table(cached_visits);
    }

    fn select(&mut self) -> usize {
        let mut current: usize = 0;
        loop {
            if self.arena.nodes[current].hot.children.is_empty() {
                return current;
            }
            current = self.get_best_child(current);
        }
    }

    fn get_best_child(&self, parent_id: usize) -> usize {
        let ids = self.arena.nodes[parent_id].hot.children.ids();
        let first = ids.start;
        debug_assert!(!ids.is_empty(), "get_best_child called on leaf node");
        if ids.len() == 1 {
            return first;
        }

        let parent_n = self.arena.nodes[parent_id].hot.n as usize;
        let child_nodes = &self.arena.nodes[ids];
        if child_nodes.len() >= 8 && child_nodes[0].hot.n == 0 {
            let offset = child_nodes
                .iter()
                .rposition(|child| child.hot.n == 0)
                .unwrap();
            return first + offset;
        }
        let parent_factor = if parent_n < self.sqrt_log.len() {
            self.sqrt_log[parent_n]
        } else {
            (parent_n as f64).ln().sqrt()
        };
        let exploration = self.c * parent_factor;
        let score = |child: &Node<S>| {
            if child.hot.n == 0 {
                f64::INFINITY
            } else {
                child.hot.q + exploration * self.inverse_sqrt[child.hot.n as usize]
            }
        };
        let mut best_offset = 0;
        let mut best_score = score(&child_nodes[0]);
        let mut offset = 1;
        let mut pairs = child_nodes[1..].chunks_exact(2);
        for pair in &mut pairs {
            let first_score = score(&pair[0]);
            let second_score = score(&pair[1]);
            if first_score.partial_cmp(&best_score).unwrap().is_ge() {
                best_offset = offset;
                best_score = first_score;
            }
            if second_score.partial_cmp(&best_score).unwrap().is_ge() {
                best_offset = offset + 1;
                best_score = second_score;
            }
            offset += 2;
        }
        if let [child] = pairs.remainder() {
            let child_score = score(child);
            if child_score.partial_cmp(&best_score).unwrap().is_ge() {
                best_offset = offset;
            }
        }
        first + best_offset
    }

    fn expand(&mut self, id: usize, legal_buf: &mut Vec<S::Action>) {
        legal_buf.clear();
        self.arena.nodes[id].state.fill_legal_actions(legal_buf);
        let parent_state: S = self.arena.nodes[id].state.clone();
        let first_child = self.arena.len();
        for &action in legal_buf.iter() {
            if S::IN_PLACE_EXPANSION {
                let child = self.arena.push(parent_state.clone(), action, Some(id));
                self.arena.nodes[child].state.step_in_place(action);
            } else {
                let state = parent_state.step(action);
                self.arena.push(state, action, Some(id));
            }
        }
        let end = self.arena.len();
        self.arena.nodes[id].hot.children = Children::from_range(first_child, end);
    }

    fn simulate<R: rand::Rng + ?Sized>(&self, id: usize, rng: &mut R) -> f64 {
        let node_state = &self.arena.nodes[id].state;
        let mut state: S = node_state.clone();
        let to_play = node_state.to_play();
        loop {
            if let Some(reward) = state.terminal_reward(to_play) {
                return reward as f64;
            }
            let action = state.get_random_legal_action(rng);
            state.step_legal_in_place(action);
        }
    }

    fn backprop(&mut self, id: usize, mut reward: f64) {
        let mut current: usize = id;
        loop {
            let hot = &mut self.arena.nodes[current].hot;
            hot.reward_sum += reward;
            hot.n += 1;
            hot.q = hot.reward_sum / hot.n as f64;
            if hot.parent == NO_PARENT {
                break;
            }
            current = hot.parent as usize;
            reward = -reward;
        }
    }
}
