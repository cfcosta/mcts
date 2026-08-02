//! The game-rules abstraction the UCT search plays through.

use rand::{seq::SliceRandom, Rng};

/// A two-player, alternating-turn game position.
///
/// The [`Mcts`](crate::Mcts) search consumes games entirely through this
/// trait: cloning positions into tree nodes, stepping them by actions,
/// and scoring finished games. Positions should be cheap to clone —
/// every rollout starts by copying one.
pub trait State {
    /// A move identifier, stored by value in every tree node.
    type Action: Copy;

    /// Whether expansion should clone into arena storage before applying
    /// `step_in_place`. The default preserves the original `step` path.
    const IN_PLACE_EXPANSION: bool = false;
    /// A placeholder action for nodes with no incoming move (the root).
    fn default_action() -> Self::Action;

    /// Whether `player` has won in this position.
    fn player_has_won(&self, player: usize) -> bool;
    /// Whether the game is over.
    fn is_terminal(&self) -> bool;
    /// Every legal action in this position.
    fn get_legal_actions(&self) -> Vec<Self::Action>;

    /// Appends every legal action to `actions`, without clearing it.
    ///
    /// The search expands through this method with one scratch buffer reused
    /// across expansions, so an implementation that pushes directly into
    /// `actions` (rather than delegating to
    /// [`get_legal_actions`](State::get_legal_actions), as the default does)
    /// removes the per-expansion allocation entirely.
    fn fill_legal_actions(&self, actions: &mut Vec<Self::Action>) {
        actions.extend(self.get_legal_actions());
    }

    /// Selects a uniformly random legal action.
    ///
    /// State implementations may override this to avoid allocating the action
    /// vector during a rollout.
    fn get_random_legal_action<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::Action {
        *self
            .get_legal_actions()
            .choose(rng)
            .expect("no legal actions in a non-terminal state")
    }

    /// The player whose turn it is.
    fn to_play(&self) -> usize;
    /// The position after `action` is played.
    fn step(&self, action: Self::Action) -> Self;

    /// Applies an action to reusable state storage.
    ///
    /// The default preserves compatibility for existing states. Implementors
    /// can override it to avoid copying a full state on every rollout ply.
    #[inline]
    fn step_in_place(&mut self, action: Self::Action)
    where
        Self: Sized,
    {
        *self = self.step(action);
    }

    /// Applies an action known to come from this state's legal-action set.
    #[inline]
    fn step_legal_in_place(&mut self, action: Self::Action)
    where
        Self: Sized,
    {
        self.step_in_place(action);
    }

    /// Returns the reward when this state is terminal, or `None` otherwise.
    #[inline]
    fn terminal_reward(&self, to_play: usize) -> Option<f32> {
        self.is_terminal().then(|| self.reward(to_play))
    }

    /// The reward of this position from `to_play`'s perspective.
    fn reward(&self, to_play: usize) -> f32;
    /// Prints the position, for debugging.
    fn render(&self);
}
