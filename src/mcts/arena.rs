//! The bump-allocated tree arena: parallel statistics and node vectors.

use bumpalo::{collections::Vec as BumpVec, Bump};

use crate::mcts::node::{Node, NodeRef, Stats, NO_PARENT};
use crate::state::State;

/// The search tree: two parallel flat vectors, related by indices.
///
/// `stats[id]` holds the visit statistics of `nodes[id]`. Splitting them
/// keeps the statistics dense — eight per cache line — so the UCB scans
/// and backpropagation walks that dominate the search touch as few lines
/// as possible, while the game states and actions stay out of their way.
///
/// Both vectors live in a caller-owned [`Bump`] arena rather than the
/// global allocator. A fresh tree is built and torn down per search, and
/// general-purpose allocators react to that rhythm — glibc in particular
/// adapts its trim and mmap thresholds to the tree size and ends up
/// returning the pages to the OS after every search and faulting them back
/// in on the next one. A bump arena reused across searches sidesteps the
/// allocator entirely: [`Bump::reset`] keeps the largest chunk, so steady-
/// state searches perform no allocator or system calls on any platform.
pub struct Arena<'b, S: State> {
    /// Visit statistics, parallel to `nodes`.
    pub stats: BumpVec<'b, Stats>,
    /// The nodes, indexed by arena id.
    pub nodes: BumpVec<'b, Node<S>>,
}

impl<'b, S: State> Arena<'b, S> {
    /// An empty arena allocating from `bump`.
    pub fn new(bump: &'b Bump) -> Self {
        Arena {
            stats: BumpVec::new_in(bump),
            nodes: BumpVec::new_in(bump),
        }
    }

    /// The number of nodes in the tree.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the tree has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub(crate) fn push(&mut self, state: S, action: S::Action, parent: Option<usize>) -> usize {
        let id = self.nodes.len();
        let parent = match parent {
            Some(parent) => u32::try_from(parent).expect("arena id exceeds u32"),
            None => NO_PARENT,
        };
        self.stats.push(Stats::new());
        self.nodes.push(Node {
            children: Default::default(),
            parent,
            state,
            action,
        });
        id
    }

    /// A read-only view of node `id`.
    pub fn get_node(&self, id: usize) -> NodeRef<'_, S> {
        let stats = &self.stats[id];
        let node = &self.nodes[id];
        NodeRef {
            state: &node.state,
            action: node.action,
            parent: match node.parent {
                NO_PARENT => None,
                parent => Some(parent as usize),
            },
            n: stats.n as usize,
            q: stats.q as f64,
            children: node.children,
        }
    }
}
