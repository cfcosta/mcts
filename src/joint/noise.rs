//! Seeded Dirichlet root noise (the
//! [`root_noise`](crate::joint::config::JointSearchConfig::root_noise)
//! extension).
//!
//! AlphaZero-style exploration noise (Silver et al., arXiv:1712.01815):
//! the root priors are blended with a Dirichlet sample so every capped
//! legal action keeps a positive probability of being explored,
//! including actions the raw priors would dismiss. All draws come from a
//! dedicated noise stream, so enabling the extension cannot shift any
//! draw on the core selection/chance/budget streams.
//!
//! Unlike those streams, whose draw semantics are frozen to one
//! `next_u64` per value in [`rng`](crate::joint::rng), the samplers here
//! use classical rejection methods — Marsaglia polar normals inside
//! Marsaglia–Tsang gammas — whose draw counts vary per sample. That is
//! safe precisely because nothing else consumes the noise stream.

use rand::RngCore;

use crate::joint::config::RootNoise;
use crate::joint::node::legal_from_priors;
use crate::joint::rng::next_f64;
use crate::joint::solver::normalized_prior;
use crate::joint::traits::Evaluation;

/// One standard normal via the Marsaglia polar method.
fn sample_normal<R: RngCore + ?Sized>(rng: &mut R) -> f64 {
    loop {
        let u = 2.0 * next_f64(rng) - 1.0;
        let v = 2.0 * next_f64(rng) - 1.0;
        let s = u * u + v * v;
        if s > 0.0 && s < 1.0 {
            return u * (-2.0 * s.ln() / s).sqrt();
        }
    }
}

/// One Gamma(alpha, 1) via Marsaglia–Tsang squeeze-and-reject, valid for
/// alpha >= 1.
fn sample_gamma_at_least_one<R: RngCore + ?Sized>(rng: &mut R, alpha: f64) -> f64 {
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (3.0 * d.sqrt());
    loop {
        let x = sample_normal(rng);
        let t = 1.0 + c * x;
        if t <= 0.0 {
            continue;
        }
        let v = t * t * t;
        let u = next_f64(rng);
        if u < 1.0 - 0.0331 * x.powi(4) {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// One Gamma(alpha, 1) draw; alpha < 1 uses the boost
/// `Gamma(alpha) = Gamma(alpha + 1) · U^(1/alpha)`.
fn sample_gamma<R: RngCore + ?Sized>(rng: &mut R, alpha: f64) -> f64 {
    debug_assert!(alpha.is_finite() && alpha > 0.0, "concentration {alpha}");
    if alpha < 1.0 {
        let boost = sample_gamma_at_least_one(rng, alpha + 1.0);
        let uniform = next_f64(rng);
        boost * uniform.powf(1.0 / alpha)
    } else {
        sample_gamma_at_least_one(rng, alpha)
    }
}

/// A symmetric Dirichlet(alpha) sample of `len` weights: normalized
/// Gamma(alpha) draws. Should every gamma underflow to zero — small
/// concentrations push `U^(1/alpha)` below the subnormal range — the
/// sample falls back to uniform rather than dividing by zero.
pub fn sample_dirichlet<R: RngCore + ?Sized>(rng: &mut R, alpha: f64, len: usize) -> Vec<f64> {
    assert!(len > 0, "cannot sample an empty Dirichlet");
    assert!(
        alpha.is_finite() && alpha > 0.0,
        "concentration must be finite and positive"
    );
    let gammas: Vec<f64> = (0..len).map(|_| sample_gamma(rng, alpha)).collect();
    let total: f64 = gammas.iter().sum();
    if total > 0.0 {
        gammas.into_iter().map(|gamma| gamma / total).collect()
    } else {
        vec![1.0 / len as f64; len]
    }
}

/// Blends Dirichlet noise into a root evaluation's priors — player side
/// first, then enemy, on the same stream.
///
/// Each side keeps the capped legal list its raw priors select, then
/// replaces the full-length priors with `(1 − ε)·normalized_prior +
/// ε·Dirichlet(alpha_scale / |legal|)` scattered over that list, zero
/// elsewhere. A side with no legal action is left untouched (node
/// construction panics on it canonically). Noise applies *before* the
/// node is built, so the stored priors, the legal-list derivation, and
/// any prior-mass pruning stacked on top all see the noised
/// distribution — preserving noise's ability to resurrect actions the
/// raw priors would prune.
pub fn apply_root_noise<R: RngCore + ?Sized>(
    evaluation: &mut Evaluation,
    player_mask: u64,
    enemy_mask: u64,
    noise: RootNoise,
    max_actions_per_side: usize,
    rng: &mut R,
) {
    noise_one_side(
        &mut evaluation.player_priors,
        player_mask,
        noise,
        max_actions_per_side,
        rng,
    );
    noise_one_side(
        &mut evaluation.enemy_priors,
        enemy_mask,
        noise,
        max_actions_per_side,
        rng,
    );
}

fn noise_one_side<R: RngCore + ?Sized>(
    priors: &mut Vec<f64>,
    mask: u64,
    noise: RootNoise,
    max_actions_per_side: usize,
    rng: &mut R,
) {
    let legal = legal_from_priors(mask, priors, max_actions_per_side);
    if legal.is_empty() {
        return;
    }
    let normalized = normalized_prior(priors, &legal);
    let eta = sample_dirichlet(rng, noise.alpha_scale / legal.len() as f64, legal.len());
    let mut noised = vec![0.0; priors.len()];
    for (index, &action) in legal.iter().enumerate() {
        noised[action] = (1.0 - noise.epsilon) * normalized[index] + noise.epsilon * eta[index];
    }
    *priors = noised;
}
