use std::{fmt, ops::Deref, ptr};

use crate::mcts::arena::Arena;
use crate::state::State;

/// An immutable child list with its single-child case stored inline.
///
/// `len <= 1` means `data` stores the child ID directly (or is unused for an
/// empty list). For larger lists, `data` is the thin data pointer of a boxed
/// slice whose length is stored in `len`.
pub struct Children {
    data: usize,
    len: usize,
}

impl Children {
    pub fn as_slice(&self) -> &[usize] {
        if self.len <= 1 {
            // `data` is an initialized, aligned usize field owned by `self`.
            unsafe { std::slice::from_raw_parts(&self.data, self.len) }
        } else {
            // Invariant: `from_range` stores a live boxed-slice data pointer
            // whenever len > 1, and Children is immutable thereafter.
            unsafe { std::slice::from_raw_parts(self.data as *const usize, self.len) }
        }
    }

    pub(crate) fn from_range(first: usize, end: usize) -> Self {
        let len = end - first;
        if len <= 1 {
            return Self { data: first, len };
        }

        let boxed = (first..end).collect::<Vec<_>>().into_boxed_slice();
        let data = Box::into_raw(boxed) as *mut usize as usize;
        Self { data, len }
    }
}

impl Default for Children {
    fn default() -> Self {
        Self { data: 0, len: 0 }
    }
}

impl Drop for Children {
    fn drop(&mut self) {
        if self.len > 1 {
            // Rebuild the exact fat pointer produced by Box::into_raw.
            let slice = ptr::slice_from_raw_parts_mut(self.data as *mut usize, self.len);
            unsafe { drop(Box::from_raw(slice)) };
        }
    }
}

impl fmt::Debug for Children {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl Deref for Children {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a> IntoIterator for &'a Children {
    type Item = &'a usize;
    type IntoIter = std::slice::Iter<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug)]
pub struct Node<S: State> {
    pub state: S,
    pub action: S::Action,
    pub parent: Option<usize>,
    pub reward_sum: f64,
    pub n: usize, // number of visits
    pub q: f64,   // average reward
    pub children: Children,
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
            children: Children::default(),
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.children.len == 0
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
