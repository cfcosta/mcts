//! Property-based tests for the seeded Dirichlet root-noise extension.
//!
//! Property inventory:
//!
//! - **Structural (unit)**: `sample_dirichlet` outputs lie on the
//!   probability simplex for any concentration, and identical seeds
//!   reproduce the sample bitwise.
//! - **Differential**: `apply_root_noise` must equal the composition of
//!   already-pinned pieces — capped `legal_from_priors`, then
//!   `normalized_prior`, then a `sample_dirichlet` blend with
//!   `alpha = alpha_scale / |legal|`, player side drawn before enemy —
//!   bitwise, replaying the same rng stream.
//! - **Structural (unit)**: noised priors form a distribution over the
//!   capped legal set — a convex blend of two distributions.
//! - **Structural (end-to-end)**: noise perturbs the stored root priors
//!   only (non-root nodes keep the evaluator's raw priors bitwise); the
//!   chance stream is isolated from noise draws (the recorded chance-seed
//!   column is position-wise identical with noise on and off); noisy
//!   searches stay bitwise deterministic under a fixed seed; and random
//!   noisy searches — with and without prior-mass pruning stacked on
//!   top — uphold every tree invariant.
//! - **Config**: the AlphaZero defaults (epsilon 0.25, alpha scale 10)
//!   and the validation envelope for both parameters.

mod support;

use hegel::generators as gs;
use hegel::TestCase;
use mcts_rs::joint::{
    apply_root_noise, legal_from_priors, normalized_prior, sample_dirichlet, Evaluation,
    JointSearchConfig, RootNoise, SearchOptions, SimultaneousTreeSearch, SplitMix64,
};
use support::joint::{
    assert_joint_tree_invariants, FixedPriorEvaluator, MatrixProvider, RecordingProvider, TwoStage,
};

// ---------------------------------------------------------------------------
// Draw helpers: valid inputs by construction, no rejection.
// ---------------------------------------------------------------------------

/// A non-empty legal mask over `n` actions.
fn draw_mask(tc: &TestCase, n: usize) -> u64 {
    tc.draw(gs::integers::<u64>().min_value(1).max_value((1 << n) - 1))
}

/// Priors in `[0, 1]`; zeros are allowed and exercise the uniform
/// fallback inside `normalized_prior`.
fn draw_priors(tc: &TestCase, n: usize) -> Vec<f64> {
    tc.draw(
        gs::vecs(gs::floats::<f64>().min_value(0.0).max_value(1.0))
            .min_size(n)
            .max_size(n),
    )
}

/// Strictly positive priors, for end-to-end runs where every action must
/// stay reachable.
fn draw_positive_priors(tc: &TestCase, n: usize) -> Vec<f64> {
    tc.draw(
        gs::vecs(gs::floats::<f64>().min_value(0.05).max_value(1.0))
            .min_size(n)
            .max_size(n),
    )
}

fn draw_matrix(tc: &TestCase, n: usize) -> Vec<f64> {
    tc.draw(
        gs::vecs(gs::floats::<f64>().min_value(-1.0).max_value(1.0))
            .min_size(n * n)
            .max_size(n * n),
    )
}

fn draw_noise(tc: &TestCase) -> RootNoise {
    RootNoise {
        epsilon: tc.draw(gs::floats::<f64>().min_value(0.05).max_value(1.0)),
        alpha_scale: tc.draw(gs::floats::<f64>().min_value(0.05).max_value(20.0)),
    }
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// The compositional reference for one side of `apply_root_noise`,
/// built purely from independently-tested pieces.
fn noised_side_reference(
    priors: &[f64],
    mask: u64,
    cap: usize,
    noise: RootNoise,
    rng: &mut SplitMix64,
) -> Vec<f64> {
    let legal = legal_from_priors(mask, priors, cap);
    if legal.is_empty() {
        return priors.to_vec();
    }
    let normalized = normalized_prior(priors, &legal);
    let eta = sample_dirichlet(rng, noise.alpha_scale / legal.len() as f64, legal.len());
    let mut noised = vec![0.0; priors.len()];
    for (index, &action) in legal.iter().enumerate() {
        noised[action] = (1.0 - noise.epsilon) * normalized[index] + noise.epsilon * eta[index];
    }
    noised
}

// ---------------------------------------------------------------------------
// Unit laws.
// ---------------------------------------------------------------------------

/// Structural oracle: a Dirichlet sample is a point on the probability
/// simplex, for concentrations well below and above 1 (both gamma
/// sampler paths).
#[hegel::test(test_cases = 256)]
fn dirichlet_samples_lie_on_the_simplex(tc: TestCase) {
    let len: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let alpha_scale: f64 = tc.draw(gs::floats::<f64>().min_value(0.05).max_value(20.0));
    let seed: u64 = tc.draw(gs::integers::<u64>());

    let sample = sample_dirichlet(&mut SplitMix64::new(seed), alpha_scale / len as f64, len);

    assert_eq!(sample.len(), len, "sample length");
    let mut total = 0.0;
    for (index, &weight) in sample.iter().enumerate() {
        assert!(
            weight.is_finite() && weight >= 0.0,
            "weight {weight} at {index}"
        );
        total += weight;
    }
    assert!((total - 1.0).abs() <= 1e-9, "total mass {total}");
}

/// Reproducibility: the sampler is a pure function of the rng state.
#[hegel::test(test_cases = 128)]
fn dirichlet_sampling_is_seed_deterministic(tc: TestCase) {
    let len: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let alpha_scale: f64 = tc.draw(gs::floats::<f64>().min_value(0.05).max_value(20.0));
    let seed: u64 = tc.draw(gs::integers::<u64>());
    let alpha = alpha_scale / len as f64;

    let first = sample_dirichlet(&mut SplitMix64::new(seed), alpha, len);
    let second = sample_dirichlet(&mut SplitMix64::new(seed), alpha, len);

    assert_eq!(bits(&first), bits(&second));
}

/// Differential oracle: `apply_root_noise` is exactly the documented
/// composition — per side, capped legal list from the raw priors,
/// normalized prior, Dirichlet noise with `alpha = alpha_scale / |legal|`
/// blended by epsilon and scattered to full length — with the player side
/// drawn before the enemy side on the same stream.
#[hegel::test(test_cases = 256)]
fn root_noise_matches_its_compositional_reference(tc: TestCase) {
    let n: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let player_mask = draw_mask(&tc, n);
    let enemy_mask = draw_mask(&tc, n);
    let player_priors = draw_priors(&tc, n);
    let enemy_priors = draw_priors(&tc, n);
    let cap: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let noise = draw_noise(&tc);
    let seed: u64 = tc.draw(gs::integers::<u64>());

    let mut evaluation = Evaluation {
        player_priors: player_priors.clone(),
        enemy_priors: enemy_priors.clone(),
        value: 0.0,
    };
    apply_root_noise(
        &mut evaluation,
        player_mask,
        enemy_mask,
        noise,
        cap,
        &mut SplitMix64::new(seed),
    );

    let mut reference_rng = SplitMix64::new(seed);
    let expected_player =
        noised_side_reference(&player_priors, player_mask, cap, noise, &mut reference_rng);
    let expected_enemy =
        noised_side_reference(&enemy_priors, enemy_mask, cap, noise, &mut reference_rng);

    assert_eq!(
        bits(&evaluation.player_priors),
        bits(&expected_player),
        "player priors"
    );
    assert_eq!(
        bits(&evaluation.enemy_priors),
        bits(&expected_enemy),
        "enemy priors"
    );
    assert_eq!(evaluation.value, 0.0, "value is untouched");
}

/// Structural oracle: the blend of two distributions over the capped
/// legal set is itself a distribution over that set — no mass leaks onto
/// actions outside it.
#[hegel::test(test_cases = 256)]
fn noised_priors_form_a_distribution_over_the_capped_legal_set(tc: TestCase) {
    let n: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let player_mask = draw_mask(&tc, n);
    let enemy_mask = draw_mask(&tc, n);
    let player_priors = draw_priors(&tc, n);
    let enemy_priors = draw_priors(&tc, n);
    let cap: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let noise = draw_noise(&tc);
    let seed: u64 = tc.draw(gs::integers::<u64>());

    let mut evaluation = Evaluation {
        player_priors: player_priors.clone(),
        enemy_priors: enemy_priors.clone(),
        value: 0.0,
    };
    apply_root_noise(
        &mut evaluation,
        player_mask,
        enemy_mask,
        noise,
        cap,
        &mut SplitMix64::new(seed),
    );

    let sides = [
        (
            "player",
            &player_priors,
            player_mask,
            &evaluation.player_priors,
        ),
        ("enemy", &enemy_priors, enemy_mask, &evaluation.enemy_priors),
    ];
    for (side, raw, mask, noised) in sides {
        let legal = legal_from_priors(mask, raw, cap);
        let mut mass = 0.0;
        for (action, &weight) in noised.iter().enumerate() {
            assert!(
                weight.is_finite() && weight >= 0.0,
                "{side}: weight {weight} at {action}"
            );
            if !legal.contains(&action) {
                assert_eq!(weight, 0.0, "{side}: mass on excluded action {action}");
            }
            mass += weight;
        }
        assert!((mass - 1.0).abs() <= 1e-9, "{side}: total mass {mass}");
    }
}

// ---------------------------------------------------------------------------
// End-to-end laws.
// ---------------------------------------------------------------------------

/// Noise applies to the root evaluation only: the stored root priors
/// differ from the evaluator's raw priors, while every deeper node keeps
/// them bitwise — and the tree stays internally coherent because the
/// stored priors are what the legal lists derive from.
#[test]
fn noise_perturbs_the_root_priors_only() {
    let player_priors = vec![0.7, 0.3];
    let enemy_priors = vec![0.6, 0.4];
    let config = JointSearchConfig {
        root_noise: Some(RootNoise::default()),
        max_depth: 2,
        expansion_budget: 16,
        regret_iterations: 64,
        ..JointSearchConfig::default()
    };
    let mut provider = TwoStage {
        stage_matrix: vec![1.0, -1.0, -1.0, 1.0],
        stage_potential: 0.25,
        bail_value: None,
    };
    let mut evaluator = FixedPriorEvaluator {
        player_priors: player_priors.clone(),
        enemy_priors: enemy_priors.clone(),
        value: 0.1,
    };
    let mut search = SimultaneousTreeSearch::new(config.clone(), 11);
    let (result, tree) = search.search_with_tree(
        &mut provider,
        &mut evaluator,
        TwoStage::root(),
        SearchOptions::default(),
    );

    assert!(result.failure.is_none(), "unexpected divergence");
    assert_joint_tree_invariants(&tree, &result, &config, "root noise");

    let root = tree.node(0);
    assert_ne!(bits(&root.player_priors), bits(&player_priors));
    assert_ne!(bits(&root.enemy_priors), bits(&enemy_priors));

    assert!(tree.nodes.len() > 1, "the deep search must create children");
    for (index, node) in tree.nodes.iter().enumerate().skip(1) {
        assert_eq!(
            bits(&node.player_priors),
            bits(&player_priors),
            "node {index} keeps raw player priors"
        );
        assert_eq!(
            bits(&node.enemy_priors),
            bits(&enemy_priors),
            "node {index} keeps raw enemy priors"
        );
    }
}

/// Stream isolation: noise draws come from their own rng stream, so the
/// chance seeds the provider receives are position-wise identical with
/// noise on and off — seeds are drawn once per sample index and shared
/// across pairs, so a noise-induced pair reorder cannot disturb the seed
/// column. The pair multiset is unchanged too: noise reorders the legal
/// lists but the full root install still covers the same grid.
#[test]
fn chance_seeds_are_isolated_from_the_noise_stream() {
    let run = |root_noise: Option<RootNoise>| {
        let config = JointSearchConfig {
            root_noise,
            chance_samples_per_joint: 2,
            expansion_budget: 1,
            regret_iterations: 32,
            ..JointSearchConfig::default()
        };
        let mut provider = RecordingProvider::new(MatrixProvider::new(3, vec![0.0; 9]));
        let root = provider.inner.root();
        let mut evaluator = FixedPriorEvaluator {
            player_priors: vec![0.5, 0.3, 0.2],
            enemy_priors: vec![0.4, 0.4, 0.2],
            value: 0.0,
        };
        let mut search = SimultaneousTreeSearch::new(config, 5);
        let result = search.search(
            &mut provider,
            &mut evaluator,
            root,
            SearchOptions::default(),
        );
        assert!(result.failure.is_none(), "unexpected divergence");
        provider.log
    };

    let log_off = run(None);
    let log_on = run(Some(RootNoise::default()));

    assert_eq!(log_off.len(), 18, "3x3 grid at two samples per pair");
    assert_eq!(log_on.len(), log_off.len());

    let seeds_off: Vec<u64> = log_off.iter().map(|entry| entry.3).collect();
    let seeds_on: Vec<u64> = log_on.iter().map(|entry| entry.3).collect();
    assert_eq!(seeds_off, seeds_on, "chance-seed column");

    let pairs = |log: &[(u64, usize, usize, u64)]| {
        let mut pairs: Vec<(usize, usize)> = log.iter().map(|entry| (entry.1, entry.2)).collect();
        pairs.sort_unstable();
        pairs
    };
    assert_eq!(pairs(&log_off), pairs(&log_on), "installed pair multiset");
}

/// Reproducibility: a noisy search is still a pure function of the seed.
#[hegel::test(test_cases = 64)]
fn noisy_searches_are_bitwise_deterministic(tc: TestCase) {
    let n: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
    let cells = draw_matrix(&tc, n);
    let player_priors = draw_positive_priors(&tc, n);
    let enemy_priors = draw_positive_priors(&tc, n);
    let value: f64 = tc.draw(gs::floats::<f64>().min_value(-1.0).max_value(1.0));
    let seed: u64 = tc.draw(gs::integers::<u64>());
    let config = JointSearchConfig {
        root_noise: Some(draw_noise(&tc)),
        expansion_budget: tc.draw(gs::integers::<u32>().min_value(1).max_value(16)),
        max_depth: tc.draw(gs::integers::<u32>().min_value(1).max_value(2)),
        regret_iterations: 32,
        ..JointSearchConfig::default()
    };
    let options = SearchOptions {
        sample_actions: tc.draw(gs::booleans()),
        router_score: 1.0,
    };

    let run = || {
        let mut provider = MatrixProvider::new(n, cells.clone());
        let root = provider.root();
        let mut evaluator = FixedPriorEvaluator {
            player_priors: player_priors.clone(),
            enemy_priors: enemy_priors.clone(),
            value,
        };
        let mut search = SimultaneousTreeSearch::new(config.clone(), seed);
        search.search(&mut provider, &mut evaluator, root, options)
    };

    let first = run();
    let second = run();
    assert_eq!(first, second);
}

/// Structural oracle: random noisy searches — optionally with prior-mass
/// pruning stacked on top, which then prunes the *noised* priors — uphold
/// every tree invariant.
#[hegel::test]
fn noisy_searches_uphold_every_tree_invariant(tc: TestCase) {
    let staged: bool = tc.draw(gs::booleans());
    let n: usize = if staged {
        2
    } else {
        tc.draw(gs::integers::<usize>().min_value(1).max_value(4))
    };
    let max_actions_per_side: usize = tc.draw(gs::integers::<usize>().min_value(1).max_value(13));
    let prune: bool = tc.draw(gs::booleans());
    let config = JointSearchConfig {
        root_noise: Some(draw_noise(&tc)),
        prior_mass_cutoff: prune.then(|| tc.draw(gs::floats::<f64>().min_value(0.05).max_value(1.0))),
        minimum_actions_per_side: tc.draw(
            gs::integers::<usize>()
                .min_value(1)
                .max_value(max_actions_per_side),
        ),
        max_actions_per_side,
        expansion_budget: tc.draw(gs::integers::<u32>().min_value(1).max_value(24)),
        minimum_expansion_budget: tc.draw(gs::integers::<u32>().min_value(1).max_value(24)),
        max_depth: tc.draw(gs::integers::<u32>().min_value(1).max_value(3)),
        chance_samples_per_joint: tc.draw(gs::integers::<u32>().min_value(1).max_value(2)),
        regret_iterations: tc.draw(gs::integers::<u32>().min_value(8).max_value(64)),
        regret_iterations_per_update: tc.draw(gs::integers::<u32>().min_value(1).max_value(16)),
        adaptive_search: tc.draw(gs::booleans()),
        ..JointSearchConfig::default()
    };
    let options = SearchOptions {
        sample_actions: tc.draw(gs::booleans()),
        router_score: tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0)),
    };
    let seed: u64 = tc.draw(gs::integers::<u64>());
    let mut evaluator = FixedPriorEvaluator {
        player_priors: draw_priors(&tc, n),
        enemy_priors: draw_priors(&tc, n),
        value: tc.draw(gs::floats::<f64>().min_value(-1.0).max_value(1.0)),
    };

    let mut search = SimultaneousTreeSearch::new(config.clone(), seed);
    let (result, tree) = if staged {
        let mut provider = TwoStage {
            stage_matrix: draw_matrix(&tc, 2),
            stage_potential: tc.draw(gs::floats::<f64>().min_value(-0.5).max_value(0.5)),
            bail_value: None,
        };
        search.search_with_tree(&mut provider, &mut evaluator, TwoStage::root(), options)
    } else {
        let mut provider = MatrixProvider::new(n, draw_matrix(&tc, n));
        let root = provider.root();
        search.search_with_tree(&mut provider, &mut evaluator, root, options)
    };

    assert!(result.failure.is_none(), "unexpected divergence");
    assert_joint_tree_invariants(&tree, &result, &config, "noisy scenario");
}

// ---------------------------------------------------------------------------
// Configuration surface.
// ---------------------------------------------------------------------------

/// The defaults follow AlphaZero (epsilon 0.25, alpha proportional to
/// 1/|legal| with scale 10 ≈ chess's 0.3 at ~35 moves), the extension is
/// off by default, and validation rejects parameters outside the
/// documented envelope. Epsilon 0 is rejected rather than treated as
/// off: renormalization dust would make it differ from `None`.
#[test]
fn noise_defaults_follow_alphazero_and_validate() {
    let noise = RootNoise::default();
    assert_eq!(noise.epsilon, 0.25);
    assert_eq!(noise.alpha_scale, 10.0);

    let config = JointSearchConfig::default();
    assert_eq!(config.root_noise, None);
    assert_eq!(config.validate(), Ok(()));

    let enabled = JointSearchConfig {
        root_noise: Some(RootNoise::default()),
        ..JointSearchConfig::default()
    };
    assert_eq!(enabled.validate(), Ok(()));

    let rejected = [
        RootNoise {
            epsilon: 0.0,
            ..RootNoise::default()
        },
        RootNoise {
            epsilon: -0.1,
            ..RootNoise::default()
        },
        RootNoise {
            epsilon: 1.5,
            ..RootNoise::default()
        },
        RootNoise {
            epsilon: f64::NAN,
            ..RootNoise::default()
        },
        RootNoise {
            alpha_scale: 0.0,
            ..RootNoise::default()
        },
        RootNoise {
            alpha_scale: -1.0,
            ..RootNoise::default()
        },
        RootNoise {
            alpha_scale: f64::INFINITY,
            ..RootNoise::default()
        },
    ];
    for noise in rejected {
        let config = JointSearchConfig {
            root_noise: Some(noise),
            ..JointSearchConfig::default()
        };
        let error = config.validate().expect_err("must reject");
        assert_eq!(error.field, "root_noise");
    }
}
