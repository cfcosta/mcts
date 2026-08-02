//! Property-based tests for the joint search, driven by hegel.
//!
//! Each property encodes a behavioral law with an oracle, not a
//! tautology:
//!
//! - **Differential**: the cold solver's reported value and
//!   exploitability are re-derived from its returned average policies by
//!   an independent naive computation; `record_value`'s running mean is
//!   checked against a naive mean; `legal_from_priors` is checked against
//!   a naive sort-and-truncate reference.
//! - **Metamorphic**: at exactly one solver iteration the time-average
//!   policy must collapse to the normalized priors — this pins the
//!   "strategy sums accumulate before the regret update" ordering.
//! - **Structural oracle**: random end-to-end searches over random
//!   providers, configurations, and seeds must uphold every check in
//!   `assert_joint_tree_invariants`, must be bitwise deterministic under
//!   a fixed seed, and must degrade to the documented fallback shape on
//!   provider divergence.
//!
//! The exact characterization suites (`joint_solver`, `joint_node`,
//! `joint_behavior`, `joint_tree_invariants`) pin the ported Python
//! semantics point-by-point; these properties sweep the input space
//! around them.

mod support;

use hegel::generators as gs;
use hegel::TestCase;
use mcts_rs::joint::{
    chance_resample_probability, legal_from_priors, mixed_policy, normalized_prior, solve_node,
    solve_zero_sum_regret, Evaluation, JointSearchConfig, RootNoise, SearchOptions, SearchResult,
    SimultaneousTreeSearch, SolverTag, Tree,
};
use support::joint::{
    assert_joint_tree_invariants, DivergeAfter, FixedPriorEvaluator, MatrixProvider,
    SeedSensitiveProvider, ToySnapshot, TwoStage,
};

// ---------------------------------------------------------------------------
// Draw helpers: valid inputs by construction, no rejection.
// ---------------------------------------------------------------------------

fn draw_matrix(tc: &TestCase, n: usize) -> Vec<f64> {
    tc.draw(
        gs::vecs(gs::floats::<f64>().min_value(-1.0).max_value(1.0))
            .min_size(n * n)
            .max_size(n * n),
    )
}

/// A non-empty legal mask over `n` actions.
fn draw_mask(tc: &TestCase, n: usize) -> u64 {
    tc.draw(gs::integers::<u64>().min_value(1).max_value((1 << n) - 1))
}

fn mask_to_list(mask: u64, n: usize) -> Vec<usize> {
    (0..n).filter(|&action| mask & (1 << action) != 0).collect()
}

/// Priors in `[0, 1]`; all-zero vectors are possible and exercise the
/// uniform fallback of `normalized_prior`.
fn draw_priors(tc: &TestCase, n: usize) -> Vec<f64> {
    tc.draw(
        gs::vecs(gs::floats::<f64>().min_value(0.0).max_value(1.0))
            .min_size(n)
            .max_size(n),
    )
}

/// Strictly positive priors, for evaluators whose legal lists must keep
/// every action reachable.
fn draw_positive_priors(tc: &TestCase, n: usize) -> Vec<f64> {
    tc.draw(
        gs::vecs(gs::floats::<f64>().min_value(0.05).max_value(1.0))
            .min_size(n)
            .max_size(n),
    )
}

fn assert_distribution(policy: &[f64], legal: &[usize], n: usize, what: &str) {
    assert_eq!(policy.len(), n, "{what}: policy length");
    let mut mass = 0.0;
    for (action, &p) in policy.iter().enumerate() {
        assert!(p >= 0.0, "{what}: negative mass {p} at action {action}");
        if !legal.contains(&action) {
            assert_eq!(p, 0.0, "{what}: mass on illegal action {action}");
        }
        mass += p;
    }
    assert!((mass - 1.0).abs() <= 1e-9, "{what}: total mass {mass}");
}

fn legal_bounds(payoff: &[f64], n: usize, p_legal: &[usize], e_legal: &[usize]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &p in p_legal {
        for &e in e_legal {
            lo = lo.min(payoff[p * n + e]);
            hi = hi.max(payoff[p * n + e]);
        }
    }
    (lo, hi)
}

/// Naive expected value of the joint policy pair, independent of the
/// solver's accumulation order.
fn reference_value(
    payoff: &[f64],
    n: usize,
    p_policy: &[f64],
    e_policy: &[f64],
    p_legal: &[usize],
    e_legal: &[usize],
) -> f64 {
    p_legal
        .iter()
        .map(|&p| {
            e_legal
                .iter()
                .map(|&e| p_policy[p] * e_policy[e] * payoff[p * n + e])
                .sum::<f64>()
        })
        .sum()
}

/// Naive best-response gap of the joint policy pair over the legal sets.
fn reference_exploitability(
    payoff: &[f64],
    n: usize,
    p_policy: &[f64],
    e_policy: &[f64],
    p_legal: &[usize],
    e_legal: &[usize],
) -> f64 {
    let best_row = p_legal
        .iter()
        .map(|&p| {
            e_legal
                .iter()
                .map(|&e| payoff[p * n + e] * e_policy[e])
                .sum::<f64>()
        })
        .fold(f64::NEG_INFINITY, f64::max);
    let best_col = e_legal
        .iter()
        .map(|&e| {
            p_legal
                .iter()
                .map(|&p| p_policy[p] * payoff[p * n + e])
                .sum::<f64>()
        })
        .fold(f64::INFINITY, f64::min);
    best_row - best_col
}

// ---------------------------------------------------------------------------
// Solver properties.
// ---------------------------------------------------------------------------

#[hegel::test(test_cases = 256)]
fn cold_solver_outputs_are_bounded_equilibrium_shaped(tc: TestCase) {
    let n: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(5));
    let payoff = draw_matrix(&tc, n);
    let p_legal = mask_to_list(draw_mask(&tc, n), n);
    let e_legal = mask_to_list(draw_mask(&tc, n), n);
    let p_priors = draw_priors(&tc, n);
    let e_priors = draw_priors(&tc, n);
    let iterations: u32 = tc.draw(gs::integers::<u32>().min_value(1).max_value(256));

    let (p, e, value, exploitability) = solve_zero_sum_regret(
        &payoff, n, &p_priors, &e_priors, &p_legal, &e_legal, iterations, false,
    );

    assert_distribution(&p, &p_legal, n, "player average");
    assert_distribution(&e, &e_legal, n, "enemy average");
    let (lo, hi) = legal_bounds(&payoff, n, &p_legal, &e_legal);
    assert!(
        value >= lo - 1e-9 && value <= hi + 1e-9,
        "value {value} outside legal payoff bounds [{lo}, {hi}]"
    );
    assert!(exploitability >= -1e-9, "exploitability {exploitability}");
}

#[hegel::test(test_cases = 256)]
fn cold_solver_reports_the_value_and_gap_of_its_own_averages(tc: TestCase) {
    let n: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(5));
    let payoff = draw_matrix(&tc, n);
    let p_legal = mask_to_list(draw_mask(&tc, n), n);
    let e_legal = mask_to_list(draw_mask(&tc, n), n);
    let p_priors = draw_priors(&tc, n);
    let e_priors = draw_priors(&tc, n);
    let iterations: u32 = tc.draw(gs::integers::<u32>().min_value(1).max_value(256));

    let (p, e, value, exploitability) = solve_zero_sum_regret(
        &payoff, n, &p_priors, &e_priors, &p_legal, &e_legal, iterations, false,
    );

    let expected_value = reference_value(&payoff, n, &p, &e, &p_legal, &e_legal);
    let expected_gap = reference_exploitability(&payoff, n, &p, &e, &p_legal, &e_legal);
    assert!(
        (value - expected_value).abs() <= 1e-9,
        "value {value} != naive {expected_value}"
    );
    assert!(
        (exploitability - expected_gap).abs() <= 1e-9,
        "exploitability {exploitability} != naive {expected_gap}"
    );
}

/// After exactly one iteration the regrets are still zero, so the only
/// strategy ever accumulated is the normalized prior — the time average
/// must equal it. This fails if the sums were accumulated after the
/// regret update instead of before.
#[hegel::test(test_cases = 256)]
fn cold_solver_single_iteration_average_is_the_normalized_prior(tc: TestCase) {
    let n: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(5));
    let payoff = draw_matrix(&tc, n);
    let p_legal = mask_to_list(draw_mask(&tc, n), n);
    let e_legal = mask_to_list(draw_mask(&tc, n), n);
    let p_priors = draw_priors(&tc, n);
    let e_priors = draw_priors(&tc, n);

    let (p, e, _, _) = solve_zero_sum_regret(
        &payoff, n, &p_priors, &e_priors, &p_legal, &e_legal, 1, false,
    );

    let expected_p = normalized_prior(&p_priors, &p_legal);
    let expected_e = normalized_prior(&e_priors, &e_legal);
    for (slot, &action) in p_legal.iter().enumerate() {
        assert!(
            (p[action] - expected_p[slot]).abs() <= 1e-12,
            "player action {action}: {} != prior {}",
            p[action],
            expected_p[slot]
        );
    }
    for (slot, &action) in e_legal.iter().enumerate() {
        assert!(
            (e[action] - expected_e[slot]).abs() <= 1e-12,
            "enemy action {action}: {} != prior {}",
            e[action],
            expected_e[slot]
        );
    }
}

#[hegel::test(test_cases = 128)]
fn warm_solve_leaves_the_node_internally_coherent(tc: TestCase) {
    let n: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(5));
    let cap: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(5));
    let config = JointSearchConfig {
        max_actions_per_side: cap,
        ..JointSearchConfig::default()
    };
    let player_mask = draw_mask(&tc, n);
    let enemy_mask = draw_mask(&tc, n);
    let evaluation = Evaluation {
        player_priors: draw_priors(&tc, n),
        enemy_priors: draw_priors(&tc, n),
        value: tc.draw(gs::floats::<f64>().min_value(-1.0).max_value(1.0)),
    };
    let iterations: u32 = tc.draw(gs::integers::<u32>().min_value(1).max_value(64));

    let mut tree: Tree<ToySnapshot> = Tree::new(n);
    let id = tree.make_node(
        ToySnapshot::live(1, player_mask, enemy_mask),
        evaluation,
        &config,
    );
    let node = tree.node_mut(id);
    let recorded: usize = tc.draw(gs::integers::<usize>().min_value(0).max_value(16));
    for _ in 0..recorded {
        let p: usize = tc.draw(gs::sampled_from(node.player_legal.clone()));
        let e: usize = tc.draw(gs::sampled_from(node.enemy_legal.clone()));
        let value: f64 = tc.draw(gs::floats::<f64>().min_value(-1.0).max_value(1.0));
        node.record_value(p, e, value);
    }
    solve_node(node, iterations, false, false);

    let node = tree.node(id);
    assert_distribution(&node.player_policy, &node.player_legal, n, "node player");
    assert_distribution(&node.enemy_policy, &node.enemy_legal, n, "node enemy");
    assert_eq!(node.solve_count, iterations, "solve count");
    let (lo, hi) = legal_bounds(&node.payoff, n, &node.player_legal, &node.enemy_legal);
    assert!(
        node.root_value >= lo - 1e-9 && node.root_value <= hi + 1e-9,
        "root value {} outside [{lo}, {hi}]",
        node.root_value
    );
    assert!(node.exploitability >= -1e-9);
    let player_sum: f64 = node.player_strategy_sum.iter().sum();
    let enemy_sum: f64 = node.enemy_strategy_sum.iter().sum();
    assert!((player_sum - f64::from(iterations)).abs() <= 1e-6);
    assert!((enemy_sum - f64::from(iterations)).abs() <= 1e-6);
    for (action, &regret) in node.player_regrets.iter().enumerate() {
        assert!(regret >= 0.0, "negative player regret at {action}");
        if !node.player_legal.contains(&action) {
            assert_eq!(regret, 0.0, "player regret on illegal action {action}");
        }
    }
    for (action, &regret) in node.enemy_regrets.iter().enumerate() {
        assert!(regret >= 0.0, "negative enemy regret at {action}");
        if !node.enemy_legal.contains(&action) {
            assert_eq!(regret, 0.0, "enemy regret on illegal action {action}");
        }
    }
}

// ---------------------------------------------------------------------------
// Pure helper properties.
// ---------------------------------------------------------------------------

#[hegel::test(test_cases = 256)]
fn record_value_tracks_the_naive_mean(tc: TestCase) {
    let values: Vec<f64> = tc.draw(
        gs::vecs(gs::floats::<f64>().min_value(-1.0).max_value(1.0))
            .min_size(1)
            .max_size(64),
    );
    let config = JointSearchConfig::default();
    let mut tree: Tree<ToySnapshot> = Tree::new(1);
    let id = tree.make_node(
        ToySnapshot::live(1, 0b1, 0b1),
        Evaluation {
            player_priors: vec![1.0],
            enemy_priors: vec![1.0],
            value: 0.0,
        },
        &config,
    );
    let node = tree.node_mut(id);
    for &value in &values {
        node.record_value(0, 0, value);
    }
    let naive = values.iter().sum::<f64>() / values.len() as f64;
    assert!(
        (node.payoff_at(0, 0) - naive).abs() <= 1e-9,
        "running mean {} != naive mean {naive}",
        node.payoff_at(0, 0)
    );
    assert_eq!(node.count_at(0, 0) as usize, values.len());
}

#[hegel::test(test_cases = 256)]
fn resample_probability_decays_monotonically_to_the_floor(tc: TestCase) {
    let floor: f64 = tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0));
    let evidence: u32 = tc.draw(gs::integers::<u32>().min_value(0).max_value(100_000));

    assert_eq!(chance_resample_probability(0, floor), 1.0);
    let here = chance_resample_probability(evidence, floor);
    let next = chance_resample_probability(evidence + 1, floor);
    assert!(here <= 1.0, "probability {here} above one");
    assert!(next <= here + 1e-12, "not monotone: {next} > {here}");
    if evidence >= 1 {
        assert!(here >= floor, "probability {here} below floor {floor}");
    }
}

/// Exploration must never starve a legal action: every legal action keeps
/// at least its epsilon-weighted share of the (possibly uniform-fallback)
/// prior, no matter how peaked the solved policy is.
#[hegel::test(test_cases = 256)]
fn mixed_policy_keeps_an_exploration_floor_under_every_legal_action(tc: TestCase) {
    let n: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
    let legal = mask_to_list(draw_mask(&tc, n), n);
    let weights = draw_positive_priors(&tc, legal.len());
    let mut policy = vec![0.0; n];
    let total: f64 = weights.iter().sum();
    for (slot, &action) in legal.iter().enumerate() {
        policy[action] = weights[slot] / total;
    }
    let all_zero: bool = tc.draw(gs::booleans());
    let priors = if all_zero {
        vec![0.0; n]
    } else {
        draw_priors(&tc, n)
    };
    let visits: u32 = tc.draw(gs::integers::<u32>().min_value(0).max_value(1000));
    let exploration: f64 = tc.draw(gs::floats::<f64>().min_value(0.0).max_value(0.5));

    let mixed = mixed_policy(&policy, &priors, &legal, visits, exploration);

    assert_distribution(&mixed, &legal, n, "mixed policy");
    let epsilon = (exploration / f64::from(visits + 1).sqrt()).max(0.02);
    let prior_share = normalized_prior(&priors, &legal);
    for (slot, &action) in legal.iter().enumerate() {
        assert!(
            mixed[action] >= epsilon * prior_share[slot] - 1e-12,
            "action {action} starved: {} < {}",
            mixed[action],
            epsilon * prior_share[slot]
        );
    }
}

#[hegel::test(test_cases = 256)]
fn legal_from_priors_matches_a_naive_sort_reference(tc: TestCase) {
    let n: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let mask = draw_mask(&tc, n);
    let priors = draw_priors(&tc, n);
    let cap: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));

    let mut expected = mask_to_list(mask, n);
    expected.sort_by(|&a, &b| priors[b].partial_cmp(&priors[a]).unwrap().then(a.cmp(&b)));
    expected.truncate(cap);

    assert_eq!(legal_from_priors(mask, &priors, cap), expected);
}

// ---------------------------------------------------------------------------
// End-to-end search properties.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum ProviderSpec {
    Matrix {
        n: usize,
        cells: Vec<f64>,
    },
    Staged {
        matrix: Vec<f64>,
        potential: f64,
        bail: Option<f64>,
    },
    SeedParity,
}

impl ProviderSpec {
    fn action_count(&self) -> usize {
        match self {
            ProviderSpec::Matrix { n, .. } => *n,
            ProviderSpec::Staged { .. } | ProviderSpec::SeedParity => 2,
        }
    }
}

#[derive(Debug, Clone)]
struct Scenario {
    spec: ProviderSpec,
    config: JointSearchConfig,
    options: SearchOptions,
    seed: u64,
    player_priors: Vec<f64>,
    enemy_priors: Vec<f64>,
    value: f64,
}

fn draw_scenario(tc: &TestCase) -> Scenario {
    let kind: &str = tc.draw(gs::sampled_from(vec!["matrix", "staged", "seed-parity"]));
    let spec = match kind {
        "matrix" => {
            let n: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
            let cells = draw_matrix(tc, n);
            ProviderSpec::Matrix { n, cells }
        }
        "staged" => {
            let matrix = draw_matrix(tc, 2);
            let potential: f64 = tc.draw(gs::floats::<f64>().min_value(-0.5).max_value(0.5));
            let bails: bool = tc.draw(gs::booleans());
            let bail = if bails {
                Some(tc.draw(gs::floats::<f64>().min_value(-1.0).max_value(1.0)))
            } else {
                None
            };
            ProviderSpec::Staged {
                matrix,
                potential,
                bail,
            }
        }
        _ => ProviderSpec::SeedParity,
    };
    let n = spec.action_count();
    let expansion_budget: u32 = tc.draw(gs::integers::<u32>().min_value(1).max_value(24));
    let max_actions_per_side: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(13));
    let prune: bool = tc.draw(gs::booleans());
    let noisy: bool = tc.draw(gs::booleans());
    let config = JointSearchConfig {
        expansion_budget,
        minimum_expansion_budget: tc.draw(gs::integers::<u32>().min_value(1).max_value(24)),
        max_depth: tc.draw(gs::integers::<u32>().min_value(1).max_value(3)),
        chance_samples_per_joint: tc.draw(gs::integers::<u32>().min_value(1).max_value(2)),
        regret_iterations: tc.draw(gs::integers::<u32>().min_value(8).max_value(128)),
        regret_iterations_per_update: tc.draw(gs::integers::<u32>().min_value(1).max_value(32)),
        deeper_joint_rotations: tc.draw(gs::integers::<usize>().min_value(1).max_value(3)),
        max_actions_per_side,
        prior_mass_cutoff: prune.then(|| tc.draw(gs::floats::<f64>().min_value(0.05).max_value(1.0))),
        minimum_actions_per_side: tc.draw(
            gs::integers::<usize>()
                .min_value(1)
                .max_value(max_actions_per_side),
        ),
        root_noise: noisy.then(|| RootNoise {
            epsilon: tc.draw(gs::floats::<f64>().min_value(0.05).max_value(1.0)),
            alpha_scale: tc.draw(gs::floats::<f64>().min_value(0.05).max_value(20.0)),
        }),
        average_strategy_policies: tc.draw(gs::booleans()),
        cfr_plus_solves: tc.draw(gs::booleans()),
        exploration: tc.draw(gs::floats::<f64>().min_value(0.0).max_value(0.5)),
        chance_resample: tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0)),
        convergence_tolerance: tc.draw(gs::floats::<f64>().min_value(0.0).max_value(0.05)),
        convergence_patience: tc.draw(gs::integers::<u32>().min_value(1).max_value(8)),
        adaptive_search: tc.draw(gs::booleans()),
        adaptive_force_deep_fraction: tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0)),
        ..JointSearchConfig::default()
    };
    let options = SearchOptions {
        sample_actions: tc.draw(gs::booleans()),
        router_score: tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0)),
    };
    Scenario {
        spec,
        config,
        options,
        seed: tc.draw(gs::integers::<u64>()),
        player_priors: draw_positive_priors(tc, n),
        enemy_priors: draw_positive_priors(tc, n),
        value: tc.draw(gs::floats::<f64>().min_value(-1.0).max_value(1.0)),
    }
}

fn run_scenario(scenario: &Scenario) -> (SearchResult, Tree<ToySnapshot>) {
    let mut evaluator = FixedPriorEvaluator {
        player_priors: scenario.player_priors.clone(),
        enemy_priors: scenario.enemy_priors.clone(),
        value: scenario.value,
    };
    let mut search = SimultaneousTreeSearch::new(scenario.config.clone(), scenario.seed);
    match &scenario.spec {
        ProviderSpec::Matrix { n, cells } => {
            let mut provider = MatrixProvider::new(*n, cells.clone());
            let root = provider.root();
            search.search_with_tree(&mut provider, &mut evaluator, root, scenario.options)
        }
        ProviderSpec::Staged {
            matrix,
            potential,
            bail,
        } => {
            let mut provider = TwoStage {
                stage_matrix: matrix.clone(),
                stage_potential: *potential,
                bail_value: *bail,
            };
            search.search_with_tree(
                &mut provider,
                &mut evaluator,
                TwoStage::root(),
                scenario.options,
            )
        }
        ProviderSpec::SeedParity => {
            let mut provider = SeedSensitiveProvider;
            search.search_with_tree(
                &mut provider,
                &mut evaluator,
                SeedSensitiveProvider::root(),
                scenario.options,
            )
        }
    }
}

#[hegel::test]
fn random_searches_uphold_every_tree_invariant(tc: TestCase) {
    let scenario = draw_scenario(&tc);
    let (result, tree) = run_scenario(&scenario);
    assert!(result.failure.is_none(), "unexpected divergence");
    assert_joint_tree_invariants(&tree, &result, &scenario.config, "hegel scenario");
}

#[hegel::test]
fn same_seed_searches_are_identical(tc: TestCase) {
    let scenario = draw_scenario(&tc);
    let (first, _) = run_scenario(&scenario);
    let (second, _) = run_scenario(&scenario);
    assert_eq!(first, second);
}

/// Divergence at a random point during the search either never fires (the
/// invariants hold) or produces the documented fallback shape.
#[hegel::test]
fn divergence_anywhere_degrades_to_the_fallback_shape(tc: TestCase) {
    let mut scenario = draw_scenario(&tc);
    let matrix = draw_matrix(&tc, 2);
    let potential: f64 = tc.draw(gs::floats::<f64>().min_value(-0.5).max_value(0.5));
    scenario.spec = ProviderSpec::Staged {
        matrix: matrix.clone(),
        potential,
        bail: None,
    };
    scenario.player_priors = draw_positive_priors(&tc, 2);
    scenario.enemy_priors = draw_positive_priors(&tc, 2);
    let successes: u32 = tc.draw(gs::integers::<u32>().min_value(0).max_value(12));

    let mut provider = DivergeAfter::new(
        TwoStage {
            stage_matrix: matrix,
            stage_potential: potential,
            bail_value: None,
        },
        successes,
    );
    let mut evaluator = FixedPriorEvaluator {
        player_priors: scenario.player_priors.clone(),
        enemy_priors: scenario.enemy_priors.clone(),
        value: scenario.value,
    };
    let mut search = SimultaneousTreeSearch::new(scenario.config.clone(), scenario.seed);
    let (result, tree) = search.search_with_tree(
        &mut provider,
        &mut evaluator,
        TwoStage::root(),
        scenario.options,
    );

    if result.failure.is_some() {
        assert_eq!(result.solver, SolverTag::DivergenceFallbackV1);
        assert_eq!(result.player_policy, vec![0.0; 2]);
        assert_eq!(result.enemy_policy, vec![0.0; 2]);
        assert_eq!(result.player_action, None);
        assert_eq!(result.enemy_action, None);
        assert!(result.payoff_matrix.is_none());
        assert_eq!(result.root_value, scenario.value);
    } else {
        assert_joint_tree_invariants(&tree, &result, &scenario.config, "hegel divergence");
    }
}
