use crate::mcts::node::{Hot, Node, NodeRef, NO_PARENT};
use crate::state::State;

/// The search tree: one flat vector of nodes, related by indices.
///
/// A single backing allocation matters here: a fresh tree is built and torn
/// down per search, and one large vector settles into memory the allocator
/// keeps warm across searches, where several parallel arrays make it return
/// the pages to the OS between trees and fault them back in on the next
/// search. The hot/cold separation lives inside each node instead, as its
/// [`Hot`] prefix.
pub struct Arena<S: State> {
    pub nodes: Vec<Node<S>>,
}

impl<S: State> Arena<S> {
    pub fn new() -> Self {
        Arena { nodes: Vec::new() }
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
