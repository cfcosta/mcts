//! Node storage: the id-span child list, the packed per-node visit
//! statistics, and the node record itself.

use std::{fmt, ops::Range};

use crate::state::State;

/// Sentinel in [`Node::parent`] for the root, which has no parent.
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

    /// The number of children.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Whether the node has no children yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl fmt::Debug for Children {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.ids()).finish()
    }
}

/// The search statistics of one node: the only data a UCB scan reads per
/// candidate and a backpropagation writes per level.
///
/// Stored in a dense array parallel to the node array (see
/// [`Arena`](crate::mcts::arena::Arena)), eight entries per cache line, so
/// scanning a contiguous child span touches an eighth of the lines that
/// full nodes would. `q` is a running mean rather than a sum that gets
/// divided on read: the mean is what every read wants, and an f32 holds it
/// exactly for the deterministic small-count cases and to ~1e-7 relative
/// everywhere else, which is far below the noise of the playouts.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Stats {
    /// Running mean reward.
    pub q: f32,
    /// Number of visits.
    pub n: u32,
}

// The scans rely on the stats record packing eight to a cache line; fail
// the build rather than silently regress if the layout grows.
const _: () = assert!(std::mem::size_of::<Stats>() == 8);

impl Stats {
    pub(crate) fn new() -> Self {
        Stats { q: 0.0, n: 0 }
    }
}

/// One node of the search tree, minus its statistics.
///
/// Everything that selection reads once per level — the child span — sits
/// in the leading bytes; the game state and action are the cold tail that
/// only expansion and rollouts touch. Visit statistics live in the arena's
/// parallel [`Stats`] array.
#[derive(Debug)]
#[repr(C)]
pub struct Node<S: State> {
    /// The node's child span; empty until expansion.
    pub children: Children,
    pub(crate) parent: u32,
    /// The game position at this node.
    pub state: S,
    /// The action that led here from the parent.
    pub action: S::Action,
}

/// A read-only view of one node.
///
/// This is the inspection API for callers and tests; the search itself
/// accesses the fields directly.
pub struct NodeRef<'a, S: State> {
    /// The node's game position.
    pub state: &'a S,
    /// The action that led here from the parent.
    pub action: S::Action,
    /// The parent's arena id; `None` for the root.
    pub parent: Option<usize>,
    /// Visit count.
    pub n: usize,
    /// Mean reward.
    pub q: f64,
    /// The node's child span.
    pub children: Children,
}
