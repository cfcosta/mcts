use crate::mcts::node::Node;
use crate::state::State;

pub struct Arena<S: State> {
    pub nodes: Vec<Node<S>>,
}

impl<S: State> Arena<S> {
    pub fn new() -> Self {
        Arena { nodes: Vec::new() }
    }

    pub fn add_node(&mut self, node: Node<S>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }

    pub(crate) fn add_child(
        &mut self,
        state: S,
        action: S::Action,
        parent: usize,
    ) -> usize {
        let id = self.nodes.len();
        if id == self.nodes.capacity() {
            self.nodes.reserve(1);
        }
        self.nodes.spare_capacity_mut()[0].write(Node::new(state, action, Some(parent)));
        // The first spare slot was initialized immediately above.
        unsafe { self.nodes.set_len(id + 1) };
        id
    }

    pub fn get_node_mut(&mut self, id: usize) -> &mut Node<S> {
        &mut self.nodes[id]
    }

    pub fn get_node(&self, id: usize) -> &Node<S> {
        &self.nodes[id]
    }
}
