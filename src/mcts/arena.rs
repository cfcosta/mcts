use bumpalo::{collections::Vec as BumpVec, Bump};

use crate::mcts::node::{Hot, Node, NodeRef, NO_PARENT};
use crate::state::State;

/// The search tree: one flat vector of nodes, related by indices.
///
/// The vector lives in a caller-owned [`Bump`] arena rather than the global
/// allocator. A fresh tree is built and torn down per search, and
/// general-purpose allocators react to that rhythm — glibc in particular
/// adapts its trim and mmap thresholds to the tree size and ends up
/// returning the pages to the OS after every search and faulting them back
/// in on the next one. A bump arena reused across searches sidesteps the
/// allocator entirely: [`Bump::reset`] keeps the largest chunk, so steady-
/// state searches perform no allocator or system calls on any platform. The
/// hot/cold separation lives inside each node, as its [`Hot`] prefix.
pub struct Arena<'b, S: State> {
    pub nodes: BumpVec<'b, Node<S>>,
}

impl<'b, S: State> Arena<'b, S> {
    pub fn new(bump: &'b Bump) -> Self {
        Arena {
            nodes: BumpVec::new_in(bump),
        }
    }

    /// The number of nodes in the tree.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub(crate) fn push(&mut self, state: S, action: S::Action, parent: Option<usize>) -> usize {
        let id = self.nodes.len();
        let parent = match parent {
            Some(parent) => u32::try_from(parent).expect("arena id exceeds u32"),
            None => NO_PARENT,
        };
        self.nodes.push(Node {
            hot: Hot::new(parent),
            state,
            action,
        });
        id
    }

    /// A read-only view of node `id`.
    pub fn get_node(&self, id: usize) -> NodeRef<'_, S> {
        let node = &self.nodes[id];
        NodeRef {
            state: &node.state,
            action: node.action,
            parent: match node.hot.parent {
                NO_PARENT => None,
                parent => Some(parent as usize),
            },
            n: node.hot.n as usize,
            q: node.hot.q,
            reward_sum: node.hot.reward_sum,
            children: node.hot.children,
        }
    }
}
