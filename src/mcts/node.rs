use std::{fmt, ops::Range};

use crate::state::State;

/// An immutable child list, stored as a span of arena ids.
///
/// Expansion pushes every child of a node onto the arena consecutively, so a
/// child list is always a contiguous id range and `(first, len)` identifies
/// it without a heap allocation.
#[derive(Clone, Copy, Default)]
pub struct Children {
    first: u32,
    len: u32,
}

impl Children {
    pub(crate) fn from_range(first: usize, end: usize) -> Self {
        Self {
            first: u32::try_from(first).expect("arena id exceeds u32"),
            len: u32::try_from(end - first).expect("child count exceeds u32"),
        }
    }

    /// The arena ids of the children, in expansion order.
    pub fn ids(&self) -> Range<usize> {
        let first = self.first as usize;
        first..first + self.len as usize
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl fmt::Debug for Children {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.ids()).finish()
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
        self.children.is_empty()
    }
}
