//! Allocation characterization for every hot path of the crate.
//!
//! A counting global allocator (thread-local, so parallel tests never see
//! each other's traffic) measures allocator calls and requested bytes
//! around each path. Three kinds of assertion:
//!
//! - **Laws** (hegel properties): allocation profiles that hold for every
//!   input — the solvers allocate a fixed number of buffers whose sizes
//!   are a closed-form function of the legal-set shape, never of the
//!   iteration count or the extension flags, the scalar helpers allocate
//!   nothing, and the scratch-threaded solve and `_into` helpers reach
//!   zero-allocation steady state once their buffers are warm.
//! - **Pins** (directed, seeded): exact counts and bytes for the four
//!   bench-mirror scenarios, so `cargo bench` timings and these numbers
//!   describe the same work.
//! - **Bands** (directed, `thread_rng`-driven): the classic UCT search is
//!   seeded from `thread_rng`, so its scratch-buffer growth varies by
//!   run; steady-state searches are asserted to stay within a small
//!   allocation band, pinning the "never touches the system allocator"
//!   claim of the arena docs to a concrete ceiling.
//!
//! Counts are source-structure facts (Vec constructions), not optimizer
//! artifacts, so measuring under the unoptimized test profile is exact
//! for the laws and pins; only the classic bands are run-dependent.

mod support;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use hegel::{generators as gs, TestCase};
use mcts_rs::joint::{
    argmax_first, average_policy, average_policy_into, chance_resample_probability, expansion_pairs,
    mixed_policy, mixed_policy_into, normalized_prior, policy_entropy, rng::next_f64, sample_index,
    solve_node, solve_node_with_scratch, solve_zero_sum_regret, strategy_weight_total, Evaluation,
    JointSearchConfig, RootNoise, SearchOptions, SimultaneousTreeSearch, SolveScratch, SplitMix64,
    Tree,
};
use mcts_rs::{Bump, Mcts};
use support::joint::{MatrixProvider, ToySnapshot, TwoStage, UniformEvaluator};
use support::{TicTacToe, WideGame};

// --- Counting allocator ---------------------------------------------------

/// Forwards to [`System`], counting each allocator call and its requested
/// bytes in thread-local cells. The cells are const-initialized (no lazy
/// setup, no destructor), so the bookkeeping itself never allocates and
/// is safe to run inside the allocator.
struct CountingAllocator;

thread_local! {
    static ALLOCATION_COUNT: Cell<u64> = const { Cell::new(0) };
    static ALLOCATED_BYTES: Cell<u64> = const { Cell::new(0) };
}

fn record(bytes: usize) {
    ALLOCATION_COUNT.set(ALLOCATION_COUNT.get() + 1);
    ALLOCATED_BYTES.set(ALLOCATED_BYTES.get() + bytes as u64);
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // A realloc is one allocator call requesting `new_size` bytes; the
        // copied-over old bytes were already counted when first requested.
        record(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Runs `work` and returns `((allocator_calls, requested_bytes), result)`
/// for the current thread. Frees are not tracked: the measurements are
/// gross allocator traffic, the quantity the arena discipline minimizes.
fn measure<T>(work: impl FnOnce() -> T) -> ((u64, u64), T) {
    let count_before = ALLOCATION_COUNT.get();
    let bytes_before = ALLOCATED_BYTES.get();
    let value = work();
    let delta = (
        ALLOCATION_COUNT.get() - count_before,
        ALLOCATED_BYTES.get() - bytes_before,
    );
    (delta, value)
}

// --- Shared fixtures (bench mirrors) --------------------------------------

/// A deterministic payoff matrix with entries in [-1, 1], exactly as the
/// criterion benches build it.
fn pseudo_matrix(action_count: usize, seed: u64) -> Vec<f64> {
    let mut rng = SplitMix64::new(seed);
    (0..action_count * action_count)
        .map(|_| next_f64(&mut rng) * 2.0 - 1.0)
        .collect()
}

/// Deterministic strictly positive unnormalized priors (bench mirror).
fn pseudo_priors(action_count: usize, seed: u64) -> Vec<f64> {
    let mut rng = SplitMix64::new(seed);
    (0..action_count)
        .map(|_| next_f64(&mut rng) + 0.05)
        .collect()
}

/// The cold solver's requested bytes: one buffer per intermediate, sized
/// by the legal-set shape alone. Mass-free prior sides change nothing —
/// the prior normalization fills its single output uniformly in place.
fn cold_solver_expected(player_len: usize, enemy_len: usize, action_count: usize) -> (u64, u64) {
    let words = player_len * enemy_len + 6 * player_len + 6 * enemy_len + 2 * action_count;
    (15, 8 * words as u64)
}

/// A node's first warm solve: eleven working buffers (two fewer than the
/// cold solver — no time-average copies) plus the two full-length
/// policies installed into the node.
fn warm_solver_first_expected(
    player_len: usize,
    enemy_len: usize,
    action_count: usize,
) -> (u64, u64) {
    let words = player_len * enemy_len + 5 * player_len + 5 * enemy_len + 2 * action_count;
    (13, 8 * words as u64)
}

/// Every later solve of the node: the installed policies are rewritten
/// in place, leaving only the eleven working buffers.
fn warm_solver_repeat_expected(player_len: usize, enemy_len: usize) -> (u64, u64) {
    let words = player_len * enemy_len + 5 * player_len + 5 * enemy_len;
    (11, 8 * words as u64)
}

/// Draws a non-empty legal subset of `0..action_count`.
fn draw_legal(tc: &TestCase, action_count: usize) -> Vec<usize> {
    let mut legal: Vec<usize> = (0..action_count)
        .filter(|_| tc.draw(gs::booleans()))
        .collect();
    if legal.is_empty() {
        legal.push(
            tc.draw(
                gs::integers::<usize>()
                    .min_value(0)
                    .max_value(action_count - 1),
            ),
        );
    }
    legal
}

fn draw_positive_priors(tc: &TestCase, action_count: usize) -> Vec<f64> {
    tc.draw(
        gs::vecs(gs::floats::<f64>().min_value(0.05).max_value(1.0))
            .min_size(action_count)
            .max_size(action_count),
    )
}

// --- Infrastructure laws --------------------------------------------------

/// The meter itself is exact: a `Vec::with_capacity` is one allocator call
/// requesting exactly `capacity * size_of::<T>()` bytes, and a closure
/// that allocates nothing measures zero.
#[hegel::test(test_cases = 100)]
fn the_meter_counts_calls_and_requested_bytes_exactly(tc: TestCase) {
    let capacity: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(4096));
    let (nothing, ()) = measure(|| ());
    assert_eq!(nothing, (0, 0), "an empty closure must measure zero");
    let (delta, buffer) = measure(|| Vec::<u64>::with_capacity(capacity));
    assert_eq!(
        delta,
        (1, 8 * capacity as u64),
        "with_capacity({capacity}) must be one exact-size call"
    );
    drop(buffer);
}

// --- Micro-helper laws ----------------------------------------------------

/// The scalar helpers on the descent path — sampling, resample decay,
/// entropy, argmax, weight totals, and the RNG itself — never allocate.
#[hegel::test(test_cases = 60)]
fn scalar_helpers_never_allocate(tc: TestCase) {
    let action_count: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let policy = draw_positive_priors(&tc, action_count);
    let legal = draw_legal(&tc, action_count);
    let evidence: u32 = tc.draw(gs::integers::<u32>().min_value(0).max_value(1_000));
    let floor: f64 = tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0));
    let solve_count: u32 = tc.draw(gs::integers::<u32>().min_value(0).max_value(1 << 20));
    let seed: u64 = tc.draw(gs::integers::<u64>());
    let mut rng = SplitMix64::new(seed);

    let (delta, _) = measure(|| sample_index(&policy, &mut rng));
    assert_eq!(delta, (0, 0), "sample_index must not allocate");
    let (delta, _) = measure(|| chance_resample_probability(evidence, floor));
    assert_eq!(
        delta,
        (0, 0),
        "chance_resample_probability must not allocate"
    );
    let (delta, _) = measure(|| policy_entropy(&policy));
    assert_eq!(delta, (0, 0), "policy_entropy must not allocate");
    let (delta, _) = measure(|| argmax_first(&policy, &legal));
    assert_eq!(delta, (0, 0), "argmax_first must not allocate");
    let (delta, _) = measure(|| {
        strategy_weight_total(false, solve_count) + strategy_weight_total(true, solve_count)
    });
    assert_eq!(delta, (0, 0), "strategy_weight_total must not allocate");
}

/// The vector helpers allocate exactly one output buffer, on every
/// branch — including the mass-free fallbacks.
#[hegel::test(test_cases = 60)]
fn vector_helpers_allocate_exactly_their_output(tc: TestCase) {
    let action_count: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let priors = draw_positive_priors(&tc, action_count);
    let policy = draw_positive_priors(&tc, action_count);
    let legal = draw_legal(&tc, action_count);
    let legal_len = legal.len() as u64;
    let visits: u32 = tc.draw(gs::integers::<u32>().min_value(0).max_value(500));
    let zero_mass: bool = tc.draw(gs::booleans());

    // normalized_prior: one legal-sized gather, renormalized in place on
    // both the positive and the mass-free uniform branch.
    let flat = vec![0.0; action_count];
    let input = if zero_mass { &flat } else { &priors };
    let (delta, _) = measure(|| normalized_prior(input, &legal));
    assert_eq!(
        delta,
        (1, 8 * legal_len),
        "normalized_prior allocation profile"
    );

    // mixed_policy: exactly the full-length output vector, on both the
    // positive-prior and uniform-fallback branches.
    let (delta, _) = measure(|| mixed_policy(&policy, input, &legal, visits, 0.1));
    assert_eq!(
        delta,
        (1, 8 * action_count as u64),
        "mixed_policy must allocate exactly its output"
    );

    // average_policy: exactly one output, whichever branch is taken.
    let weight: f64 = if zero_mass { 0.0 } else { 1.0 };
    let (delta, _) = measure(|| average_policy(&priors, weight, &policy));
    assert_eq!(
        delta,
        (1, 8 * action_count as u64),
        "average_policy must allocate exactly its output"
    );
}

/// Full-grid expansion pairs are one exact-capacity buffer; the rotation
/// path grows by pushes and must match a plain push-loop of the same
/// final length exactly — no hidden intermediates.
#[hegel::test(test_cases = 60)]
fn expansion_pairs_allocate_like_a_plain_push_loop(tc: TestCase) {
    let action_count: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let player_legal = draw_legal(&tc, action_count);
    let enemy_legal = draw_legal(&tc, action_count);
    let rotations: usize = tc.draw(gs::integers::<usize>().min_value(0).max_value(4));

    let full_cells = (player_legal.len() * enemy_legal.len()) as u64;
    let (delta, _) = measure(|| expansion_pairs(&player_legal, &enemy_legal, true, rotations));
    assert_eq!(
        delta,
        (1, 16 * full_cells),
        "the full grid must be one exact-capacity buffer"
    );

    let (delta, pairs) = measure(|| expansion_pairs(&player_legal, &enemy_legal, false, rotations));
    let (reference, buffer) = measure(|| {
        let mut reference: Vec<(usize, usize)> = Vec::new();
        for index in 0..pairs.len() {
            reference.push((index, index));
        }
        reference
    });
    assert_eq!(
        delta,
        reference,
        "rotation pairs must allocate exactly like pushing {} pairs",
        pairs.len()
    );
    drop(buffer);
}

// --- Solver laws ----------------------------------------------------------

/// The cold solver's allocation profile is a closed-form function of the
/// legal-set shape: fifteen buffers, `8·(P·E + 6P + 6E + 2n)` bytes —
/// for every payoff, every prior (mass-free sides included), every
/// iteration count, and both CFR+ settings. Iterating more costs zero
/// allocator traffic.
#[hegel::test(test_cases = 40)]
fn cold_solver_allocations_depend_only_on_the_legal_shape(tc: TestCase) {
    let action_count: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
    let payoff = tc.draw(
        gs::vecs(gs::floats::<f64>().min_value(-1.0).max_value(1.0))
            .min_size(action_count * action_count)
            .max_size(action_count * action_count),
    );
    let zero_player: bool = tc.draw(gs::booleans());
    let zero_enemy: bool = tc.draw(gs::booleans());
    let flat = vec![0.0; action_count];
    let player_priors = if zero_player {
        flat.clone()
    } else {
        draw_positive_priors(&tc, action_count)
    };
    let enemy_priors = if zero_enemy {
        flat
    } else {
        draw_positive_priors(&tc, action_count)
    };
    let player_legal = draw_legal(&tc, action_count);
    let enemy_legal = draw_legal(&tc, action_count);
    let low_iterations: u32 = tc.draw(gs::integers::<u32>().min_value(1).max_value(8));
    let high_iterations: u32 = tc.draw(gs::integers::<u32>().min_value(9).max_value(48));

    let expected = cold_solver_expected(player_legal.len(), enemy_legal.len(), action_count);
    for iterations in [low_iterations, high_iterations] {
        for cfr_plus in [false, true] {
            let (delta, _) = measure(|| {
                solve_zero_sum_regret(
                    &payoff,
                    action_count,
                    &player_priors,
                    &enemy_priors,
                    &player_legal,
                    &enemy_legal,
                    iterations,
                    cfr_plus,
                )
            });
            assert_eq!(
                delta, expected,
                "cold solve at {iterations} iterations (cfr_plus: {cfr_plus}) \
                 must match the shape formula"
            );
        }
    }
}

/// The warm node solve's allocation profile is a closed-form function of
/// the node shape alone: the first solve pays thirteen buffers,
/// `8·(P·E + 5P + 5E + 2n)` bytes, and every later solve rewrites the
/// installed policies in place and pays only the eleven working buffers
/// — independent of the iteration count and of both extension flags.
#[hegel::test(test_cases = 40)]
fn warm_solver_allocations_depend_only_on_the_node_shape(tc: TestCase) {
    let action_count: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
    let player_legal = draw_legal(&tc, action_count);
    let enemy_legal = draw_legal(&tc, action_count);
    let player_mask: u64 = player_legal.iter().map(|&action| 1u64 << action).sum();
    let enemy_mask: u64 = enemy_legal.iter().map(|&action| 1u64 << action).sum();
    let low_iterations: u32 = tc.draw(gs::integers::<u32>().min_value(1).max_value(8));
    let high_iterations: u32 = tc.draw(gs::integers::<u32>().min_value(9).max_value(48));
    let leaf_value: f64 = tc.draw(gs::floats::<f64>().min_value(-1.0).max_value(1.0));

    let config = JointSearchConfig::default();
    let mut tree: Tree<ToySnapshot> = Tree::new(action_count);
    let node_id = tree.make_node(
        ToySnapshot::live(0, player_mask, enemy_mask),
        Evaluation {
            player_priors: draw_positive_priors(&tc, action_count),
            enemy_priors: draw_positive_priors(&tc, action_count),
            value: leaf_value,
        },
        &config,
    );
    let node = tree.node_mut(node_id);
    let first_expected = warm_solver_first_expected(
        node.player_legal.len(),
        node.enemy_legal.len(),
        action_count,
    );
    let repeat_expected =
        warm_solver_repeat_expected(node.player_legal.len(), node.enemy_legal.len());

    let mut solves = 0u32;
    for iterations in [low_iterations, high_iterations, low_iterations] {
        for (average_policies, cfr_plus) in
            [(false, false), (false, true), (true, false), (true, true)]
        {
            let (delta, ()) =
                measure(|| solve_node(&mut *node, iterations, average_policies, cfr_plus));
            let expected = if solves == 0 {
                first_expected
            } else {
                repeat_expected
            };
            solves += 1;
            assert_eq!(
                delta, expected,
                "warm solve #{solves} at {iterations} iterations (average: \
                 {average_policies}, cfr_plus: {cfr_plus}) must match the shape formula"
            );
        }
    }
}

/// A caller-held scratch makes every solve after a node's first
/// completely allocation-free: the scratch buffers and the installed
/// policies are all rewritten in place, whatever the iteration count or
/// flag combination. A second same-shaped node entering the warm scratch
/// pays exactly its own two first-install policies, nothing else.
#[hegel::test(test_cases = 40)]
fn scratch_solves_reach_zero_allocation_steady_state(tc: TestCase) {
    let action_count: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
    let player_legal = draw_legal(&tc, action_count);
    let enemy_legal = draw_legal(&tc, action_count);
    let player_mask: u64 = player_legal.iter().map(|&action| 1u64 << action).sum();
    let enemy_mask: u64 = enemy_legal.iter().map(|&action| 1u64 << action).sum();
    let iterations: u32 = tc.draw(gs::integers::<u32>().min_value(1).max_value(24));

    let config = JointSearchConfig::default();
    let mut tree: Tree<ToySnapshot> = Tree::new(action_count);
    let first_id = tree.make_node(
        ToySnapshot::live(0, player_mask, enemy_mask),
        Evaluation {
            player_priors: draw_positive_priors(&tc, action_count),
            enemy_priors: draw_positive_priors(&tc, action_count),
            value: 0.0,
        },
        &config,
    );
    let second_id = tree.make_node(
        ToySnapshot::live(1, player_mask, enemy_mask),
        Evaluation {
            player_priors: draw_positive_priors(&tc, action_count),
            enemy_priors: draw_positive_priors(&tc, action_count),
            value: 0.0,
        },
        &config,
    );
    let mut scratch = SolveScratch::default();

    let node = tree.node_mut(first_id);
    let (first, ()) =
        measure(|| solve_node_with_scratch(&mut *node, iterations, false, false, &mut scratch));
    assert_eq!(
        first,
        warm_solver_first_expected(player_legal.len(), enemy_legal.len(), action_count),
        "the first scratch solve pays the full first-solve profile"
    );
    for (average_policies, cfr_plus) in [(false, false), (false, true), (true, false), (true, true)] {
        let (delta, ()) = measure(|| {
            solve_node_with_scratch(
                &mut *node,
                iterations,
                average_policies,
                cfr_plus,
                &mut scratch,
            )
        });
        assert_eq!(
            delta,
            (0, 0),
            "repeat scratch solve (average: {average_policies}, cfr_plus: {cfr_plus}) \
             must not allocate"
        );
    }

    let node = tree.node_mut(second_id);
    let (delta, ()) =
        measure(|| solve_node_with_scratch(&mut *node, iterations, false, false, &mut scratch));
    assert_eq!(
        delta,
        (2, 16 * action_count as u64),
        "a fresh same-shaped node pays exactly its two installed policies"
    );
    let (delta, ()) =
        measure(|| solve_node_with_scratch(&mut *node, iterations, false, false, &mut scratch));
    assert_eq!(delta, (0, 0), "and is allocation-free from then on");
}

/// The `_into` policy helpers pay exactly one exact-sized allocation
/// while their output buffer is cold and nothing once it is warm — on
/// every branch, including the average fallback.
#[hegel::test(test_cases = 60)]
fn into_helpers_reach_zero_allocations_on_warm_buffers(tc: TestCase) {
    let action_count: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let priors = draw_positive_priors(&tc, action_count);
    let policy = draw_positive_priors(&tc, action_count);
    let strategy_sum = draw_positive_priors(&tc, action_count);
    let legal = draw_legal(&tc, action_count);
    let visits: u32 = tc.draw(gs::integers::<u32>().min_value(0).max_value(500));
    let weight: f64 = tc.draw(gs::floats::<f64>().min_value(-1.0).max_value(64.0));

    let mut buffer: Vec<f64> = Vec::new();
    let (delta, ()) =
        measure(|| mixed_policy_into(&policy, &priors, &legal, visits, 0.1, &mut buffer));
    assert_eq!(
        delta,
        (1, 8 * action_count as u64),
        "cold mixed_policy_into pays exactly its output"
    );
    let (delta, ()) =
        measure(|| mixed_policy_into(&policy, &priors, &legal, visits, 0.1, &mut buffer));
    assert_eq!(delta, (0, 0), "warm mixed_policy_into must not allocate");
    let (delta, ()) = measure(|| average_policy_into(&strategy_sum, weight, &policy, &mut buffer));
    assert_eq!(
        delta,
        (0, 0),
        "warm average_policy_into must not allocate (weight {weight})"
    );
}

// --- Bench-mirror pins (deterministic joint paths) ------------------------

/// joint/cold_solve_13x13_2048: the root equilibrium.
const COLD_SOLVE_13X13_2048: (u64, u64) = (15, 2808);
/// joint/warm_solve_16_13x13: the per-learned-simulation node solve. A
/// node's very first solve additionally installs its two policies; the
/// bench loop's steady state is the repeat profile.
const WARM_SOLVE_16_13X13_FIRST: (u64, u64) = (13, 2600);
const WARM_SOLVE_16_13X13_REPEAT: (u64, u64) = (11, 2392);
/// joint/root_only_13: a full root-only search call (169-cell install
/// plus cold equilibrium plus result assembly).
const ROOT_ONLY_13: (u64, u64) = (251, 94_724);
/// joint/deep_two_stage_budget_320: a full deep search at the default
/// transition budget — descent, resampling, convergence tracking, and
/// one warm node solve per learned simulation. The engine-held solver
/// scratch and mix buffers absorb every per-solve and per-descent
/// buffer (this pin was (10_519, 228_836) with per-call vectors),
/// leaving node construction, first-install policies, the two cold
/// equilibria, and result assembly.
const DEEP_TWO_STAGE_BUDGET_320: (u64, u64) = (379, 59_204);
/// The deep bench scenario with every opt-in extension stacked on
/// (prior-mass cutoff, root noise, average-strategy policies, CFR+);
/// (10_310, 227_492) before the scratch. The extensions add no
/// per-solve buffers, only a different trajectory over the same budget.
const DEEP_TWO_STAGE_ALL_EXTENSIONS: (u64, u64) = (396, 61_540);

#[test]
fn cold_solve_bench_mirror_matches_its_pin() {
    let n = 13;
    let payoff = pseudo_matrix(n, 1);
    let player_priors = pseudo_priors(n, 2);
    let enemy_priors = pseudo_priors(n, 3);
    let legal: Vec<usize> = (0..n).collect();
    let (delta, _) = measure(|| {
        solve_zero_sum_regret(
            &payoff,
            n,
            &player_priors,
            &enemy_priors,
            &legal,
            &legal,
            2048,
            false,
        )
    });
    assert_eq!(delta, COLD_SOLVE_13X13_2048, "joint/cold_solve_13x13_2048");
    assert_eq!(
        delta,
        cold_solver_expected(n, n, n),
        "the pin must agree with the shape formula"
    );
}

#[test]
fn warm_solve_bench_mirror_matches_its_pin() {
    let n = 13;
    let mask = (1u64 << n) - 1;
    let config = JointSearchConfig::default();
    let mut tree: Tree<ToySnapshot> = Tree::new(n);
    let node_id = tree.make_node(
        ToySnapshot::live(0, mask, mask),
        Evaluation {
            player_priors: pseudo_priors(n, 2),
            enemy_priors: pseudo_priors(n, 3),
            value: 0.0,
        },
        &config,
    );
    let node = tree.node_mut(node_id);
    let values = pseudo_matrix(n, 4);
    for player in 0..n {
        for enemy in 0..n {
            node.record_value(player, enemy, values[player * n + enemy]);
        }
    }
    let (first, ()) = measure(|| solve_node(&mut *node, 16, false, false));
    let (second, ()) = measure(|| solve_node(&mut *node, 16, false, false));
    assert_eq!(
        first, WARM_SOLVE_16_13X13_FIRST,
        "joint/warm_solve_16_13x13 first solve"
    );
    assert_eq!(
        second, WARM_SOLVE_16_13X13_REPEAT,
        "joint/warm_solve_16_13x13 steady state"
    );
    assert_eq!(
        first,
        warm_solver_first_expected(n, n, n),
        "the first pin must agree with the shape formula"
    );
    assert_eq!(
        second,
        warm_solver_repeat_expected(n, n),
        "the repeat pin must agree with the shape formula"
    );
}

/// Measures one seeded search call, everything constructed outside the
/// window, and asserts a repeat run is allocation-identical (the whole
/// path is deterministic given the seed).
fn measure_root_only_search() -> (u64, u64) {
    let n = 13;
    let matrix = pseudo_matrix(n, 5);
    let config = JointSearchConfig {
        expansion_budget: 1,
        minimum_expansion_budget: 1,
        ..JointSearchConfig::default()
    };
    let mut provider = MatrixProvider::new(n, matrix);
    let mut evaluator = UniformEvaluator {
        action_count: n,
        value: 0.0,
    };
    let mut search = SimultaneousTreeSearch::new(config, 17);
    let root = provider.root();
    let (delta, result) = measure(|| {
        search.search(
            &mut provider,
            &mut evaluator,
            root,
            SearchOptions::default(),
        )
    });
    assert!(result.failure.is_none(), "the mirror search must succeed");
    delta
}

#[test]
fn root_only_search_matches_its_pin() {
    let first = measure_root_only_search();
    let second = measure_root_only_search();
    assert_eq!(first, ROOT_ONLY_13, "joint/root_only_13");
    assert_eq!(second, first, "same-seed searches are allocation-identical");
}

fn measure_deep_two_stage(config: JointSearchConfig) -> (u64, u64) {
    let mut provider = TwoStage {
        stage_matrix: vec![1.0, -1.0, -1.0, 1.0],
        stage_potential: 0.15,
        bail_value: Some(-1.0),
    };
    let mut evaluator = UniformEvaluator {
        action_count: 2,
        value: -0.2,
    };
    let mut search = SimultaneousTreeSearch::new(config, 23);
    let root = TwoStage::root();
    let (delta, result) = measure(|| {
        search.search(
            &mut provider,
            &mut evaluator,
            root,
            SearchOptions::default(),
        )
    });
    assert!(result.failure.is_none(), "the mirror search must succeed");
    delta
}

#[test]
fn deep_search_matches_its_pin() {
    let config = JointSearchConfig {
        max_depth: 3,
        ..JointSearchConfig::default()
    };
    let first = measure_deep_two_stage(config.clone());
    let second = measure_deep_two_stage(config);
    assert_eq!(
        first, DEEP_TWO_STAGE_BUDGET_320,
        "joint/deep_two_stage_budget_320"
    );
    assert_eq!(second, first, "same-seed searches are allocation-identical");
}

#[test]
fn deep_search_with_every_extension_matches_its_pin() {
    let config = JointSearchConfig {
        max_depth: 3,
        prior_mass_cutoff: Some(0.5),
        root_noise: Some(RootNoise::default()),
        average_strategy_policies: true,
        cfr_plus_solves: true,
        ..JointSearchConfig::default()
    };
    let first = measure_deep_two_stage(config.clone());
    let second = measure_deep_two_stage(config);
    assert_eq!(first, DEEP_TWO_STAGE_ALL_EXTENSIONS, "extensions stacked");
    assert_eq!(second, first, "same-seed searches are allocation-identical");
}

// --- Classic UCT bands (thread_rng-driven, run-dependent) -----------------

/// Steady state for the classic search: warmed lookup tables, a `Bump`
/// retained across searches sized by a larger warm-up run. The measured
/// search may then touch the system allocator only to grow its two
/// per-search scratch buffers (`path`, `legal_buf`).
fn measure_steady_classic<S: mcts_rs::State + std::fmt::Debug + Clone>(
    state: S,
    c: f64,
    iterations: usize,
) -> (u64, u64) {
    let bump = Bump::new();
    // Two double-length warm-ups: the retained chunk ends up larger than
    // any tree the measured search can build, and the shared lookup
    // tables plus this thread's rng are initialized on the way.
    for _ in 0..2 {
        Mcts::new(&bump, state.clone(), c).search(2 * iterations);
    }
    let mut bump = bump;
    bump.reset();
    let (delta, _) = measure(|| Mcts::new(&bump, state, c).search(iterations));
    delta
}

#[test]
fn steady_state_tic_tac_toe_search_touches_only_scratch_buffers() {
    // Observed sample: 4 calls, 130 bytes for the whole 1000-iteration
    // search — the growth chains of `path` and `legal_buf`, nothing else.
    let (count, bytes) = measure_steady_classic(TicTacToe::new(), 0.5, 1_000);
    assert!(
        count <= 16 && bytes <= 4_096,
        "steady TTT search(1000) must stay within the scratch-buffer band, \
         measured ({count}, {bytes})"
    );
}

#[test]
fn steady_state_wide_search_touches_only_scratch_buffers() {
    // Observed sample: 3 calls, 20,496 bytes — `legal_buf` growing to the
    // 512-action width dominates; the arena itself stays silent.
    let (count, bytes) = measure_steady_classic(WideGame::new(512), 1.0, 1_024);
    assert!(
        count <= 16 && bytes <= 65_536,
        "steady WideGame(512) search(1024) must stay within the \
         scratch-buffer band, measured ({count}, {bytes})"
    );
}

#[test]
fn cold_tree_build_allocates_only_bump_chunks() {
    // Warm the lookup tables and this thread's rng, then measure a search
    // on a fresh, empty Bump: the traffic is the arena's chunk chain plus
    // the scratch buffers. Observed sample: 15 calls, 527,842 bytes — the
    // cost `Bump::reset` reuse saves on every later search.
    let warm_bump = Bump::new();
    Mcts::new(&warm_bump, TicTacToe::new(), 0.5).search(2_000);
    drop(warm_bump);
    let bump = Bump::new();
    let ((count, bytes), _) = measure(|| Mcts::new(&bump, TicTacToe::new(), 0.5).search(1_000));
    assert!(
        count <= 64 && bytes <= 2 << 20,
        "cold TTT search(1000) must stay within the chunk-chain band, \
         measured ({count}, {bytes})"
    );
}
