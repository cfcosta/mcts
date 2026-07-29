use std::{fmt, ops::Range};

use crate::state::State;

/// Sentinel in [`Hot::parent`] for the root, which has no parent.
pub(crate) const NO_PARENT: u32 = u32::MAX;

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

/// The hot part of a node: everything a UCB scan reads and a backpropagation
/// writes — visit statistics, parent link, and child span — packed into
/// exactly 32 bytes.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Hot {
    pub q: f64, // average reward
    pub reward_sum: f64,
    pub n: u32, // number of visits
    pub(crate) parent: u32,
    pub children: Children,
}

// The scans rely on the hot record spanning half a cache line; fail the
// build rather than silently regress if the layout grows.
const _: () = assert!(std::mem::size_of::<Hot>() == 32);

impl Hot {
    pub(crate) fn new(parent: u32) -> Self {
        Hot {
            q: 0.0,
            reward_sum: 0.0,
            n: 0,
            parent,
            children: Children::default(),
        }
    }
}

/// One node of the search tree.
///
/// `repr(C)` pins the hot record to the leading bytes, so selection and
/// backpropagation only ever touch the first part of each node while the
/// game state and action stay in the cold tail.
#[derive(Debug)]
#[repr(C)]
pub struct Node<S: State> {
    pub hot: Hot,
    pub state: S,
    pub action: S::Action,
}

/// A read-only view of one node.
///
/// This is the inspection API for callers and tests; the search itself
/// accesses the fields directly.
pub struct NodeRef<'a, S: State> {
    pub state: &'a S,
    pub action: S::Action,
    pub parent: Option<usize>,
    pub n: usize,
    pub q: f64,
    pub reward_sum: f64,
    pub children: Children,
}
