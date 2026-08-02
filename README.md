# Rust Monte Carlo Tree Search (MCTS) with Arena Allocator

[![Crates.io](https://img.shields.io/crates/v/mcts-rs.svg)](https://crates.io/crates/mcts-rs)
[![Documentation](https://docs.rs/mcts-rs/badge.svg)](https://docs.rs/mcts-rs)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Rust](https://img.shields.io/badge/rust-1.56%2B-orange.svg)

A Rust implementation of the Monte Carlo Tree Search (MCTS) algorithm using an arena allocator for efficient memory management. This project features a Tic-Tac-Toe game to showcase the MCTS algorithm in action.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Adding to Your Project](#adding-to-your-project)
  - [Running the Tic-Tac-Toe Example](#running-the-tic-tac-toe-example)
  - [Testing and Benchmarking](#testing-and-benchmarking)
- [Implementation Details](#implementation-details)
  - [Arena Allocator](#arena-allocator)
  - [MCTS Algorithm](#mcts-algorithm)
  - [State Trait](#state-trait)
  - [Joint Simultaneous-Move Search](#joint-simultaneous-move-search)
- [License](#license)

## Introduction

Monte Carlo Tree Search (MCTS) is a search algorithm used for decision-making processes. This project provides a Rust implementation of MCTS that efficiently manages memory using an arena allocator. By storing all nodes in a central arena, we avoid the overhead of reference counting and interior mutability, resulting in a more performant and idiomatic Rust codebase.

The `mcts-rs` crate is now available on [crates.io](https://crates.io/crates/mcts-rs), making it easy to include in your Rust projects.

The included Tic-Tac-Toe game serves as a practical example of how to use the MCTS library.

## Features

- **Efficient Memory Management**: Utilizes an arena allocator to store nodes, reducing allocation overhead.
- **Generic State Management**: Defines a `State` trait to allow MCTS to work with any game or decision process.

## Getting Started

### Prerequisites

- **Rust**: Ensure you have Rust and Cargo installed. You can install Rust using [rustup](https://rustup.rs/).

### Adding to Your Project

To include `mcts-rs` in your project, add the following to your `Cargo.toml`:

```toml
[dependencies]
mcts-rs = "0.1.0"
```

Replace `"0.1.0"` with the latest version available on [crates.io](https://crates.io/crates/mcts-rs).

### Running the Tic-Tac-Toe Example

To run the Tic-Tac-Toe game where the MCTS algorithm plays against itself with a random starting move, clone the repository and use the following commands:

```bash
git clone https://github.com/PaytonWebber/mcts-rs.git
cd mcts-rs
cargo run --example tic_tac_toe
```

### Testing and Benchmarking

The repository ships with a behavioral test suite (`cargo test`) and a
Criterion benchmark suite (`cargo bench`) intended to make performance work
safe: the tests pin the observable behavior of the search, and the benchmarks
isolate each hot path. See [BENCHMARKING.md](BENCHMARKING.md) for the
workflow, and [CACHE_ANALYSIS.md](CACHE_ANALYSIS.md) for a measured
analysis of the search's cache behavior.

## Implementation Details

### Arena Allocator

An arena allocator is used to efficiently manage memory for the nodes in the MCTS tree. All nodes live in a single flat vector and relationships are represented by indices rather than pointers or references. Each node leads with a packed 32-byte hot record — visit statistics, parent link, and child span — so the selection and backpropagation loops only touch the leading bytes of every node they scan, while the game state and action stay in the cold tail. This approach avoids the need for reference counting (`Rc`) and interior mutability (`RefCell`), leading to cleaner and more efficient code.

The node vector itself is allocated from a caller-owned [`bumpalo`](https://docs.rs/bumpalo) bump arena rather than the global allocator. Keep one `Bump` alive across searches and reset it between them, and steady-state searches never touch the system allocator — general-purpose allocators (glibc in particular) otherwise return the tree's pages to the OS after every search and fault them back in on the next one:

```rust
use mcts_rs::{Bump, Mcts};

let mut bump = Bump::new();
while !game.is_terminal() {
    let action = Mcts::new(&bump, game.clone(), 1.4).search(1_000);
    game = game.step(action);
    bump.reset(); // keeps the arena's memory for the next search
}
```

**Benefits:**

- **Performance**: Reduced allocation overhead and improved cache locality.
- **Simplicity**: Simplifies ownership and borrowing by avoiding complex lifetime issues.
- **Safety**: Leverages Rust's safety guarantees without resorting to unsafe code.

### MCTS Algorithm

The MCTS algorithm consists of four main steps:

1. **Selection**: Starting from the root node, select child nodes based on the Upper Confidence Bound (UCB) until a leaf node is reached.
2. **Expansion**: If the leaf node is not a terminal state, expand it by adding all possible child nodes.
3. **Simulation**: Run a simulation from the expanded node to a terminal state by making random moves.
4. **Backpropagation**: Update the nodes along the path with the simulation result.

**Key Components:**

- **Node Building Blocks** (`node.rs`): The `Node` layout with its packed `Hot` prefix, the `Children` id span, and the read-only `NodeRef` view of a single node.
- **Arena Struct** (`arena.rs`): Stores the tree as one flat vector of nodes in a caller-provided bump arena.
- **MCTS Implementation** (`mod.rs`): Contains the logic for selection, expansion, simulation, and backpropagation.

### State Trait

The `State` trait abstracts the game logic, allowing the MCTS algorithm to work with any game or decision process that implements this trait. Here's the trait definition:

```rust
pub trait State {
    /// The type of action that can be taken in the state (e.g., tuple of coordinates). 
    type Action: Copy;
    
    /// Returns the default action for the state (used for root node).
    fn default_action() -> Self::Action;

    /// Checks if the specified player has won the game.
    fn player_has_won(&self, player: usize) -> bool;

    /// Determines if the current state is a terminal state (no further moves possible).
    fn is_terminal(&self) -> bool;

    /// Returns a vector of legal actions available from the current state.
    fn get_legal_actions(&self) -> Vec<Self::Action>;

    /// Returns the index of the player whose turn it is to play.
    fn to_play(&self) -> usize;

    /// Returns a new state resulting from applying the given action to the current state.
    fn step(&self, action: Self::Action) -> Self;
    
    /// Calculates and returns the reward for the specified player in the current state.
    fn reward(&self, player: usize) -> f32;

    /// Renders or prints the current state (useful for debugging or display purposes).
    fn render(&self);
}
```

By implementing this trait for your game or decision process, you can integrate it with the MCTS algorithm provided in this library. The `examples/tic_tac_toe.rs` and `examples/ultimate_tic_tac_toe.rs` examples offer complete, self-contained implementations of the `State` trait, including the optional fast paths for rollouts and in-place expansion.

### Joint Simultaneous-Move Search

The `joint` module is a second, independent search for **simultaneous-move
zero-sum games**: both players commit an action at once, and the outcome may
depend on hidden chance. It is a port of a Pokémon battle pipeline's search
and shares nothing with the UCT implementation above — turn order, priors,
seeded chance, and matrix solves have no representation in the `State` trait.

Instead of visit counts and UCB, every tree node carries a dense payoff
matrix over joint `(player, enemy)` action pairs, filled by chance-sampled
transitions and solved with **regret matching+**: warm-started incremental
solves during the descent, and a cold 2048-iteration equilibrium at the
root. Descent samples joint actions from epsilon-mixed policies, resamples
chance outcomes with decaying probability, shapes leaf values with a
potential function, and stops on root-policy convergence, budget
exhaustion, or starved learning. An optional adaptive router decides per
search whether the tree descent is worth its cost or the root equilibrium
alone suffices.

Integration happens through three traits in `joint::traits`:

- **`JointSnapshot`**: a game state — identity, per-side legal-action
  bitmasks, terminal value, and an optional potential for value shaping.
- **`TransitionProvider`**: steps a snapshot by a joint action pair and a
  chance seed; may report a `Divergence`, which the search converts into a
  well-formed fallback result instead of panicking.
- **`Evaluator`**: produces per-side policy priors and a value estimate
  (typically a neural network in the real pipeline; fixed tables in the
  tests).

Searches are driven by `joint::SimultaneousTreeSearch`, are fully
deterministic for a given seed, and return a `SearchResult` with policies,
sampled or argmax actions, the root payoff matrix, and rich diagnostics.
See the `src/joint/mod.rs` module documentation for the algorithm details
and the list of deliberate deviations from the Python source, and
`benches/joint.rs` (`cargo bench --bench joint`) for benchmarks of the
solver and end-to-end search hot paths.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
