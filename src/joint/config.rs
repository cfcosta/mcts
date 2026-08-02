//! Search hyper-parameters, mirroring the Python `JointSearchConfig`
//! dataclass field for field (defaults included). Two fields are
//! deliberately absent: `inference_batch_size` (a batching concern of the
//! Python pipeline) and `redundant_action_prior_scale` (pre-search prior
//! shaping the caller applies before handing priors to the search).

use std::fmt;

/// Parameters of the seeded Dirichlet root-noise extension
/// ([`JointSearchConfig::root_noise`]).
///
/// AlphaZero mixes `(1 − ε)·prior + ε·Dirichlet(α)` into the root priors
/// so every root action keeps a positive exploration probability (Silver
/// et al., arXiv:1712.01815). The concentration follows their inverse
/// scaling with the move count — `α = alpha_scale / |legal|` per side —
/// and the defaults reproduce their constants: ε = 0.25, and a scale of
/// 10 giving α ≈ 0.3 at chess's ~35 legal moves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootNoise {
    /// Noise share of the blended prior, validated to (0, 1]. Zero is
    /// rejected rather than treated as off: renormalization would still
    /// perturb the priors, so `None` is the only exact off switch.
    pub epsilon: f64,
    /// Numerator of the per-side concentration `alpha_scale / |legal|`,
    /// validated to be finite and positive.
    pub alpha_scale: f64,
}

impl Default for RootNoise {
    fn default() -> Self {
        Self {
            epsilon: 0.25,
            alpha_scale: 10.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct JointSearchConfig {
    /// Chance outcomes sampled per joint action pair during expansion.
    pub chance_samples_per_joint: u32,
    /// Tree depth limit; 1 solves the root matrix only.
    pub max_depth: u32,
    /// Iterations of the cold root equilibrium solve.
    pub regret_iterations: u32,
    /// Cap on legal actions kept per side, highest-prior first.
    pub max_actions_per_side: usize,
    /// Transition budget for the descent phase (the root install is exempt).
    pub expansion_budget: u32,
    /// Numerator of the visit-decayed epsilon in the mixed descent policy.
    pub exploration: f64,
    /// Floor probability of resampling a fresh chance outcome.
    pub chance_resample: f64,
    /// RM+ iterations applied per warm node update.
    pub regret_iterations_per_update: u32,
    /// Diagonal rotations used when expanding non-root nodes partially.
    pub deeper_joint_rotations: usize,
    /// Transitions required before convergence may stop the search early.
    pub minimum_expansion_budget: u32,
    /// L1 policy-change threshold counted toward the convergence streak.
    pub convergence_tolerance: f64,
    /// Consecutive stable updates required to declare convergence.
    pub convergence_patience: u32,
    /// Enables the deep/shallow routing predicates.
    pub adaptive_search: bool,
    /// Router score at or above which the search always goes deep.
    pub adaptive_deep_threshold: f64,
    /// Probability of forcing a deep search as a calibration sample.
    pub adaptive_force_deep_fraction: f64,
    /// Root online exploitability at or above which the search goes deep.
    pub adaptive_exploitability_threshold: f64,
    /// Root payoff spread at or above which (with enough policy entropy)
    /// the search goes deep.
    pub adaptive_payoff_spread_threshold: f64,
    /// Opt-in extension: keep only each side's highest-prior prefix
    /// holding this share of the raw prior mass. `None` (the default)
    /// disables the cutoff and keeps the full capped legal lists.
    pub prior_mass_cutoff: Option<f64>,
    /// Fewest actions per side the mass cutoff may keep; consulted only
    /// when `prior_mass_cutoff` is set.
    pub minimum_actions_per_side: usize,
    /// Opt-in extension: blend seeded Dirichlet noise into the root
    /// priors before the root node is built. `None` (the default) leaves
    /// the evaluator's priors untouched.
    pub root_noise: Option<RootNoise>,
    /// Opt-in extension: warm node solves install the cumulative
    /// time-average strategy (`strategy_sum / solve_count`) instead of
    /// the ported last iterate, with node value and exploitability
    /// recomputed on the averages. `false` (the default) keeps the
    /// Python last-iterate behavior.
    pub average_strategy_policies: bool,
    /// Opt-in extension: every RM+ solve — warm node solves and the cold
    /// root equilibrium — runs with CFR+'s accelerations (Tammelin,
    /// arXiv:1407.5042): alternating regret updates and linearly
    /// weighted strategy averaging, with warm nodes continuing the
    /// linear weights globally across batches. `false` (the default)
    /// keeps the ported simultaneous uniform-average dynamics bitwise.
    pub cfr_plus_solves: bool,
    /// Opt-in extension: the cold root equilibria stop early once their
    /// time-average exploitability, checked every
    /// [`EQUILIBRIUM_CHECK_INTERVAL`](crate::joint::solver::EQUILIBRIUM_CHECK_INTERVAL)
    /// iterations, is at most this bound — solving to a target
    /// exploitability instead of a fixed iteration count, as CFR+
    /// deployments do (Tammelin, arXiv:1407.5042; Bowling et al.,
    /// Science 2015). A stopped solve is bit-identical to a
    /// `regret_iterations`-capped solve truncated at the stopping
    /// checkpoint, and the performed count is surfaced in
    /// [`RootDiagnostics::equilibrium_iterations`](crate::joint::result::RootDiagnostics::equilibrium_iterations).
    /// `None` (the default) always runs the full `regret_iterations`
    /// bitwise.
    pub equilibrium_tolerance: Option<f64>,
}

impl Default for JointSearchConfig {
    fn default() -> Self {
        Self {
            chance_samples_per_joint: 1,
            max_depth: 1,
            regret_iterations: 2048,
            max_actions_per_side: 13,
            expansion_budget: 320,
            exploration: 0.10,
            chance_resample: 0.10,
            regret_iterations_per_update: 16,
            deeper_joint_rotations: 2,
            minimum_expansion_budget: 192,
            convergence_tolerance: 0.005,
            convergence_patience: 8,
            adaptive_search: false,
            adaptive_deep_threshold: 0.55,
            adaptive_force_deep_fraction: 0.10,
            adaptive_exploitability_threshold: 0.08,
            adaptive_payoff_spread_threshold: 0.75,
            prior_mass_cutoff: None,
            minimum_actions_per_side: 2,
            root_noise: None,
            average_strategy_policies: false,
            cfr_plus_solves: false,
            equilibrium_tolerance: None,
        }
    }
}

/// A config field whose value violates an assumption the search relies on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConfigError {
    pub field: &'static str,
    pub requirement: &'static str,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} must be {}", self.field, self.requirement)
    }
}

impl std::error::Error for ConfigError {}

impl JointSearchConfig {
    /// Checks every assumption the search makes of its configuration.
    ///
    /// `SimultaneousTreeSearch::new` panics on the first violation; this
    /// method exists separately so the rules stay table-testable.
    pub fn validate(&self) -> Result<(), ConfigError> {
        fn positive(value: bool, field: &'static str) -> Result<(), ConfigError> {
            if value {
                Ok(())
            } else {
                Err(ConfigError {
                    field,
                    requirement: "at least 1",
                })
            }
        }
        positive(
            self.chance_samples_per_joint >= 1,
            "chance_samples_per_joint",
        )?;
        positive(self.max_depth >= 1, "max_depth")?;
        positive(self.regret_iterations >= 1, "regret_iterations")?;
        positive(self.max_actions_per_side >= 1, "max_actions_per_side")?;
        positive(self.expansion_budget >= 1, "expansion_budget")?;
        positive(
            self.regret_iterations_per_update >= 1,
            "regret_iterations_per_update",
        )?;
        positive(self.deeper_joint_rotations >= 1, "deeper_joint_rotations")?;
        positive(self.convergence_patience >= 1, "convergence_patience")?;

        fn finite_non_negative(value: f64, field: &'static str) -> Result<(), ConfigError> {
            if value.is_finite() && value >= 0.0 {
                Ok(())
            } else {
                Err(ConfigError {
                    field,
                    requirement: "finite and non-negative",
                })
            }
        }
        finite_non_negative(self.exploration, "exploration")?;
        finite_non_negative(self.convergence_tolerance, "convergence_tolerance")?;

        fn unit_interval(value: f64, field: &'static str) -> Result<(), ConfigError> {
            if value.is_finite() && (0.0..=1.0).contains(&value) {
                Ok(())
            } else {
                Err(ConfigError {
                    field,
                    requirement: "within [0, 1]",
                })
            }
        }
        unit_interval(self.chance_resample, "chance_resample")?;
        unit_interval(
            self.adaptive_force_deep_fraction,
            "adaptive_force_deep_fraction",
        )?;

        fn finite(value: f64, field: &'static str) -> Result<(), ConfigError> {
            if value.is_finite() {
                Ok(())
            } else {
                Err(ConfigError {
                    field,
                    requirement: "finite",
                })
            }
        }
        finite(self.adaptive_deep_threshold, "adaptive_deep_threshold")?;
        finite(
            self.adaptive_exploitability_threshold,
            "adaptive_exploitability_threshold",
        )?;
        finite(
            self.adaptive_payoff_spread_threshold,
            "adaptive_payoff_spread_threshold",
        )?;

        if let Some(cutoff) = self.prior_mass_cutoff {
            if !(cutoff.is_finite() && cutoff > 0.0 && cutoff <= 1.0) {
                return Err(ConfigError {
                    field: "prior_mass_cutoff",
                    requirement: "within (0, 1]",
                });
            }
            positive(
                self.minimum_actions_per_side >= 1,
                "minimum_actions_per_side",
            )?;
            if self.minimum_actions_per_side > self.max_actions_per_side {
                return Err(ConfigError {
                    field: "minimum_actions_per_side",
                    requirement: "at most max_actions_per_side",
                });
            }
        }
        if let Some(noise) = self.root_noise {
            if !(noise.epsilon.is_finite() && noise.epsilon > 0.0 && noise.epsilon <= 1.0) {
                return Err(ConfigError {
                    field: "root_noise",
                    requirement: "an epsilon within (0, 1]",
                });
            }
            if !(noise.alpha_scale.is_finite() && noise.alpha_scale > 0.0) {
                return Err(ConfigError {
                    field: "root_noise",
                    requirement: "a finite positive alpha_scale",
                });
            }
        }
        if let Some(tolerance) = self.equilibrium_tolerance {
            if !(tolerance.is_finite() && tolerance > 0.0) {
                return Err(ConfigError {
                    field: "equilibrium_tolerance",
                    requirement: "finite and positive",
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_python_defaults() {
        let config = JointSearchConfig::default();
        assert_eq!(config.chance_samples_per_joint, 1);
        assert_eq!(config.max_depth, 1);
        assert_eq!(config.regret_iterations, 2048);
        assert_eq!(config.max_actions_per_side, 13);
        assert_eq!(config.expansion_budget, 320);
        assert_eq!(config.exploration, 0.10);
        assert_eq!(config.chance_resample, 0.10);
        assert_eq!(config.regret_iterations_per_update, 16);
        assert_eq!(config.deeper_joint_rotations, 2);
        assert_eq!(config.minimum_expansion_budget, 192);
        assert_eq!(config.convergence_tolerance, 0.005);
        assert_eq!(config.convergence_patience, 8);
        assert!(!config.adaptive_search);
        assert_eq!(config.adaptive_deep_threshold, 0.55);
        assert_eq!(config.adaptive_force_deep_fraction, 0.10);
        assert_eq!(config.adaptive_exploitability_threshold, 0.08);
        assert_eq!(config.adaptive_payoff_spread_threshold, 0.75);
        assert_eq!(config.validate(), Ok(()));
    }

    #[test]
    fn extensions_default_off_and_skip_their_validation() {
        let config = JointSearchConfig::default();
        assert_eq!(config.prior_mass_cutoff, None);
        assert_eq!(config.minimum_actions_per_side, 2);
        assert_eq!(config.root_noise, None);
        assert!(!config.average_strategy_policies);
        assert!(!config.cfr_plus_solves);
        assert_eq!(config.equilibrium_tolerance, None);

        // The floor only binds while the cutoff is enabled: existing
        // configs with small action caps stay valid.
        let unpruned = JointSearchConfig {
            max_actions_per_side: 1,
            ..JointSearchConfig::default()
        };
        assert_eq!(unpruned.validate(), Ok(()));
    }

    type Mutation = fn(&mut JointSearchConfig);

    #[test]
    fn validate_reports_the_offending_field() {
        let cases: [(&str, Mutation); 24] = [
            ("chance_samples_per_joint", |c| {
                c.chance_samples_per_joint = 0
            }),
            ("max_depth", |c| c.max_depth = 0),
            ("regret_iterations", |c| c.regret_iterations = 0),
            ("max_actions_per_side", |c| c.max_actions_per_side = 0),
            ("expansion_budget", |c| c.expansion_budget = 0),
            ("regret_iterations_per_update", |c| {
                c.regret_iterations_per_update = 0
            }),
            ("deeper_joint_rotations", |c| c.deeper_joint_rotations = 0),
            ("convergence_patience", |c| c.convergence_patience = 0),
            ("exploration", |c| c.exploration = -0.1),
            ("convergence_tolerance", |c| {
                c.convergence_tolerance = f64::NAN
            }),
            ("chance_resample", |c| c.chance_resample = 1.5),
            ("adaptive_force_deep_fraction", |c| {
                c.adaptive_force_deep_fraction = -0.2
            }),
            ("adaptive_deep_threshold", |c| {
                c.adaptive_deep_threshold = f64::INFINITY
            }),
            ("adaptive_exploitability_threshold", |c| {
                c.adaptive_exploitability_threshold = f64::NAN
            }),
            ("adaptive_payoff_spread_threshold", |c| {
                c.adaptive_payoff_spread_threshold = f64::NEG_INFINITY
            }),
            ("prior_mass_cutoff", |c| c.prior_mass_cutoff = Some(0.0)),
            ("prior_mass_cutoff", |c| {
                c.prior_mass_cutoff = Some(f64::NAN)
            }),
            ("minimum_actions_per_side", |c| {
                c.prior_mass_cutoff = Some(0.5);
                c.minimum_actions_per_side = 0;
            }),
            ("minimum_actions_per_side", |c| {
                c.prior_mass_cutoff = Some(0.5);
                c.minimum_actions_per_side = c.max_actions_per_side + 1;
            }),
            ("root_noise", |c| {
                c.root_noise = Some(RootNoise {
                    epsilon: 0.0,
                    ..RootNoise::default()
                })
            }),
            ("root_noise", |c| {
                c.root_noise = Some(RootNoise {
                    alpha_scale: f64::NAN,
                    ..RootNoise::default()
                })
            }),
            ("equilibrium_tolerance", |c| {
                c.equilibrium_tolerance = Some(0.0)
            }),
            ("equilibrium_tolerance", |c| {
                c.equilibrium_tolerance = Some(-0.01)
            }),
            ("equilibrium_tolerance", |c| {
                c.equilibrium_tolerance = Some(f64::NAN)
            }),
        ];
        for (field, mutate) in cases {
            let mut config = JointSearchConfig::default();
            mutate(&mut config);
            let error = config.validate().expect_err(field);
            assert_eq!(error.field, field);
        }
    }
}
