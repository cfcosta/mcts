pub mod arena;
pub mod node;

use crate::state::State;
use arena::Arena;
use node::{Children, Stats};

use bumpalo::Bump;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

fn shared_lookup_table<T: Copy>(
    tables: &'static OnceLock<Mutex<HashMap<usize, &'static [T]>>>,
    required: usize,
    value: fn(usize) -> T,
) -> &'static [T] {
    let len = required
        .saturating_add(1)
        .checked_next_power_of_two()
        .unwrap();
    let tables = tables.get_or_init(|| Mutex::new(HashMap::new()));
    let mut tables = tables.lock().unwrap();
    tables.entry(len).or_insert_with(|| {
        let values = (0..len).map(value).collect::<Vec<_>>().into_boxed_slice();
        Box::leak(values)
    })
}

// f32 like the statistics it scores: sixteen entries per cache line, and
// no widening of the child's q on the hot scan path.
fn inverse_sqrt_table(required: usize) -> &'static [f32] {
    static TABLES: OnceLock<Mutex<HashMap<usize, &'static [f32]>>> = OnceLock::new();
    shared_lookup_table(&TABLES, required, |n| {
        if n == 0 {
            f32::INFINITY
        } else {
            1.0 / (n as f32).sqrt()
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
    inverse_sqrt: &'static [f32],
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
            inverse_sqrt: &[f32::INFINITY],
            sqrt_log: &[0.0],
        }
    }

    pub fn search(&mut self, n: usize) -> S::Action {
        let current_visits = self.arena.stats[self.root_id].n as usize;
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
        // The nodes visited by the current descent, root first. Recording
        // them lets backpropagation replay the path instead of chasing
        // parent pointers, where every step's load depends on the previous
        // one. Reused across iterations like `legal_buf`.
        let mut path: Vec<u32> = Vec::new();

        for _ in 0..n {
            path.clear();
            let mut selected_id: usize = self.select(&mut path);
            if !self.arena.nodes[selected_id].state.is_terminal() {
                self.expand(selected_id, &mut legal_buf);
                let children = self.arena.nodes[selected_id].children.ids();
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
                path.push(selected_id as u32);
            }
            let reward: f64 = self.simulate(selected_id, &mut rng);
            self.backprop(&path, reward);
        }
        let best_child: usize = self.arena.nodes[self.root_id]
            .children
            .ids()
            .max_by(|&a, &b| {
                let node_a_score = self.arena.stats[a].q;
                let node_b_score = self.arena.stats[b].q;
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

    /// Descends from the root to a leaf, appending every visited node
    /// (including both ends) to `path`.
    fn select(&mut self, path: &mut Vec<u32>) -> usize {
        let mut current: usize = 0;
        loop {
            path.push(current as u32);
            if self.arena.nodes[current].children.is_empty() {
                return current;
            }
            current = self.get_best_child(current);
        }
    }

    fn get_best_child(&self, parent_id: usize) -> usize {
        let ids = self.arena.nodes[parent_id].children.ids();
        let first = ids.start;
        debug_assert!(!ids.is_empty(), "get_best_child called on leaf node");
        if ids.len() == 1 {
            return first;
        }

        let parent_n = self.arena.stats[parent_id].n as usize;
        let child_stats = &self.arena.stats[ids];
        if child_stats.len() >= 8 && child_stats[0].n == 0 {
            let offset = child_stats.iter().rposition(|child| child.n == 0).unwrap();
            return first + offset;
        }
        let parent_factor = if parent_n < self.sqrt_log.len() {
            self.sqrt_log[parent_n]
        } else {
            (parent_n as f64).ln().sqrt()
        };
        // The scan itself runs entirely in f32 — the precision of the
        // statistics being compared; only the once-per-scan parent factor
        // is computed in f64.
        let exploration = (self.c * parent_factor) as f32;
        let score = |child: &Stats| {
            if child.n == 0 {
                f32::INFINITY
            } else {
                child.q + exploration * self.inverse_sqrt[child.n as usize]
            }
        };
        if child_stats.len() >= 8 {
            return first + Self::argmax_wide(child_stats, score);
        }
        let mut best_offset = 0;
        let mut best_score = score(&child_stats[0]);
        let mut offset = 1;
        let mut pairs = child_stats[1..].chunks_exact(2);
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

    /// The offset of the best-scoring entry, resolving ties toward the
    /// highest offset — the same winner the sequential `is_ge` scan picks.
    ///
    /// Eight independent running maxima (one per lane of a fixed-width
    /// chunk) replace the single running maximum, so the comparisons carry
    /// no loop-to-loop dependency: the lanes pipeline instead of waiting
    /// on each other, and the compiler is free to pack them into vector
    /// registers where the target allows.
    fn argmax_wide(child_stats: &[Stats], score: impl Fn(&Stats) -> f32) -> usize {
        let mut lane_best = [f32::NEG_INFINITY; 8];
        let mut lane_offset = [0u32; 8];
        let mut base = 0u32;
        let mut chunks = child_stats.chunks_exact(8);
        for chunk in &mut chunks {
            for lane in 0..8 {
                let lane_score = score(&chunk[lane]);
                if lane_score >= lane_best[lane] {
                    lane_best[lane] = lane_score;
                    lane_offset[lane] = base + lane as u32;
                }
            }
            base += 8;
        }
        // Within a lane, `>=` already kept the last achiever; across lanes
        // the last achiever of the overall maximum is the largest offset.
        let mut best_score = f32::NEG_INFINITY;
        let mut best_offset = 0u32;
        for lane in 0..8 {
            if lane_best[lane] > best_score
                || (lane_best[lane] == best_score && lane_offset[lane] > best_offset)
            {
                best_score = lane_best[lane];
                best_offset = lane_offset[lane];
            }
        }
        for (i, child) in chunks.remainder().iter().enumerate() {
            if score(child) >= best_score {
                best_score = score(child);
                best_offset = base + i as u32;
            }
        }
        best_offset as usize
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
        self.arena.nodes[id].children = Children::from_range(first_child, end);
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

    /// Applies `reward` (from the last path node's perspective) to every
    /// node on `path`, flipping sign per level. Walking the recorded path
    /// leaf-first touches the same nodes in the same order as following
    /// parent links did, but the ids are all known up front, so the stat
    /// updates are independent loads instead of a serial pointer chase.
    /// Each mean is updated incrementally — `q += (reward - q) / n` is the
    /// mean of all `n` rewards, with no stored sum.
    fn backprop(&mut self, path: &[u32], reward: f64) {
        let mut reward = reward as f32;
        for &id in path.iter().rev() {
            let stats = &mut self.arena.stats[id as usize];
            stats.n += 1;
            stats.q += (reward - stats.q) / stats.n as f32;
            reward = -reward;
        }
    }
}
