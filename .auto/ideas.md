# Promising optimization ideas

- Remove cached `legal_actions` vectors from bundled states and generate actions without retaining per-node heap buffers; then consider a zero-allocation internal action iteration API for rollouts.
- Split hot node metadata from cold state/action storage (SoA or compact hot struct) while preserving the public `Arena::nodes`/`Node` API expected by downstream users.
- Replace per-node `children: Vec<usize>` allocation with a contiguous child range, if this can be done without breaking the existing public representation contract.
- Reuse rollout scratch state/action storage through an optional State API that remains efficient for external implementations.
- Add realistic arena capacity planning or chunked storage after measuring growth/copy tradeoffs across small and large searches.
