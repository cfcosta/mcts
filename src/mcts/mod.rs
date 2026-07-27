pub mod arena;
pub mod node;

use crate::state::State;
use arena::Arena;
use node::{Children, Node};

use rand::seq::SliceRandom;

pub struct Mcts<S: State> {
    pub arena: Arena<S>,
    pub root_id: usize,
    c: f64,
    inverse_sqrt: Vec<f64>,
    sqrt_log: Vec<f64>,
}

impl<S: State + std::fmt::Debug + std::clone::Clone> Mcts<S> {
    pub fn new(state: S, c: f64) -> Self {
        let mut arena: Arena<S> = Arena::new();
        let root: Node<S> = Node::new(state.clone(), S::default_action(), None);
        let root_id: usize = arena.add_node(root);
        Mcts {
            arena,
            root_id,
            c,
            inverse_sqrt: vec![f64::INFINITY],
            sqrt_log: vec![0.0],
        }
    }

    pub fn search(&mut self, n: usize) -> S::Action {
        let current_visits = self.arena.get_node(self.root_id).n;
        let cached_visits = current_visits.saturating_add(n);
        let mut inverse_ready = self.inverse_sqrt.len() > cached_visits;
        let mut factors_ready = self.sqrt_log.len() > cached_visits;
        let mut rng = rand::thread_rng();

        for _ in 0..n {
            let mut selected_id: usize = self.select();
            let selected_node: &Node<S> = self.arena.get_node(selected_id);
            if !selected_node.state.is_terminal() {
                self.expand(selected_id);
                let child_count = self.arena.get_node(selected_id).children.len();
                if child_count > 1 {
                    if !inverse_ready {
                        self.extend_inverse_sqrt(cached_visits);
                        inverse_ready = true;
                    }
                    // Parent factors are most useful when a long search or a
                    // very wide scan amortizes their eager construction.
                    if !factors_ready && (cached_visits >= 4_096 || child_count >= 256) {
                        self.extend_sqrt_log(cached_visits);
                        factors_ready = true;
                    }
                }
                let children = &self.arena.get_node(selected_id).children;
                let random_child: usize = children.choose(&mut rng).unwrap().clone();
                selected_id = random_child;
            }
            let reward: f64 = self.simulate(selected_id, &mut rng);
            self.backprop(selected_id, reward);
        }
        let root_node: &Node<S> = self.arena.get_node(self.root_id);
        let best_child: usize = root_node
            .children
            .iter()
            .max_by(|&a, &b| {
                let node_a_score = self.arena.get_node(*a).q;
                let node_b_score = self.arena.get_node(*b).q;
                node_a_score.partial_cmp(&node_b_score).unwrap()
            })
            .unwrap()
            .clone();

        let best_action: S::Action = self.arena.get_node(best_child).action;
        best_action
    }

    #[cold]
    #[inline(never)]
    fn extend_inverse_sqrt(&mut self, cached_visits: usize) {
        self.inverse_sqrt.extend(
            (self.inverse_sqrt.len()..=cached_visits).map(|n| 1.0 / (n as f64).sqrt()),
        );
    }

    #[cold]
    #[inline(never)]
    fn extend_sqrt_log(&mut self, cached_visits: usize) {
        self.sqrt_log.extend(
            (self.sqrt_log.len()..=cached_visits).map(|n| (n as f64).ln().sqrt()),
        );
    }

    fn select(&mut self) -> usize {
        let mut current: usize = 0;
        loop {
            let node = &self.arena.get_node(current);
            if node.is_leaf() {
                return current;
            }
            current = self.get_best_child(current);
        }
    }

    fn get_best_child(&self, parent_id: usize) -> usize {
        let parent = self.arena.get_node(parent_id);
        let (&first, rest) = parent
            .children
            .split_first()
            .expect("get_best_child called on leaf node");
        if rest.is_empty() {
            return first;
        }

        debug_assert!(
            rest.iter()
                .enumerate()
                .all(|(offset, &id)| id == first + offset + 1),
            "expanded children must remain contiguous"
        );
        let child_nodes = &self.arena.nodes[first..first + parent.children.len()];
        let parent_factor = if parent.n < self.sqrt_log.len() {
            self.sqrt_log[parent.n]
        } else {
            (parent.n as f64).ln().sqrt()
        };
        let exploration = self.c * parent_factor;
        let score = |child: &Node<S>| {
            if child.n == 0 {
                f64::INFINITY
            } else {
                child.q + exploration * self.inverse_sqrt[child.n]
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

    fn expand(&mut self, id: usize) {
        let parent: &Node<S> = self.arena.get_node_mut(id);
        let legal_actions: Vec<S::Action> = parent.state.get_legal_actions();
        let parent_state: S = parent.state.clone();
        let first_child = self.arena.nodes.len();
        for action in legal_actions {
            let child = self.arena.add_child(parent_state.clone(), action, id);
            self.arena
                .get_node_mut(child)
                .state
                .step_in_place(action);
        }
        let end = self.arena.nodes.len();
        self.arena.get_node_mut(id).children = Children::from_range(first_child, end);
    }

    fn simulate<R: rand::Rng + ?Sized>(&self, id: usize, rng: &mut R) -> f64 {
        let node: &Node<S> = self.arena.get_node(id);
        let mut state: S = node.state.clone();
        let to_play = node.state.to_play();
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
            let node = self.arena.get_node_mut(current);
            node.reward_sum += reward;
            node.n += 1;
            node.q = node.reward_sum / node.n as f64;
            if let Some(parent_id) = node.parent {
                current = parent_id;
            } else {
                break;
            }
            reward = -reward;
        }
    }
}
