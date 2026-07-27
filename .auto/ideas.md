# Promising optimization ideas

- A breaking `Node`/`Arena` redesign could split hot `{parent, n, q, reward_sum, child range}` metadata from cold state/action storage. The duplicating sidecar prototype improved deep-chain ~9% but hurt TTT; a true source-of-truth SoA should avoid that tradeoff.
- Replace boxed child-ID lists with `{first_child, len}` in a future major API revision. Contiguity is verified, but current public tests/API require slice-like borrowed IDs.
- Add a caller-configurable RNG API. A per-search `SmallRng` improved UTTT 6-7% and deep-chain 14% but regressed TTT under the current default; opt-in RNG ownership could expose the speed without changing defaults.
- Explore packed UTTT storage using two `[u64; 2]` bitboards with metadata in unused bits. This could shrink state from 44 to 32 bytes and UTTT nodes from 104 toward 96 bytes, but variable 9-bit extraction may offset cache gains.
- Consider global/tiered immutable UCB lookup tables for applications constructing many Mcts instances. Per-search tables win strongly now; sharing could remove repeated setup in full games without penalizing small one-off searches if tiered carefully.
