use crate::mcts::arena::Arena;
use crate::state::State;

#[derive(Debug)]
pub struct Node<S: State> {
    pub state: S,
    pub action: S::Action,
    pub parent: Option<usize>,
    pub reward_sum: f64,
    pub n: usize, // number of visits
    pub q: f64,   // average reward
    pub children: Vec<usize>,
}

impl<S: State> Node<S> {
    pub fn new(state: S, action: S::Action, parent: Option<usize>) -> Self {
        Node {
            state,
            action,
            parent,
            reward_sum: 0.0,
            n: 0,
            q: 0.0,
            children: Vec::new(),
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub fn ucb(&self, arena: &Arena<S>, c: f64) -> f64 {
        if self.n == 0 {
            return f64::INFINITY;
        }
        let parent_n = arena.get_node(self.parent.unwrap()).n as f64;
        self.q + c * (parent_n.ln() / self.n as f64).sqrt()
    }

    pub fn get_best_child(&self, arena: &Arena<S>, c: f64) -> usize {
        let (&first, rest) = self
            .children
            .split_first()
            .expect("get_best_child called on leaf node");
        if rest.is_empty() {
            return first;
        }

        let parent_log = (self.n as f64).ln();
        let score = |id| {
            let child = arena.get_node(id);
            if child.n == 0 {
                f64::INFINITY
            } else {
                child.q + c * (parent_log / child.n as f64).sqrt()
            }
        };
        let mut best_child = first;
        let mut best_score = score(first);
        for &child in rest {
            let child_score = score(child);
            // Iterator::max_by returns the last element when scores tie.
            if child_score.partial_cmp(&best_score).unwrap().is_ge() {
                best_child = child;
                best_score = child_score;
            }
        }
        best_child
    }
}
