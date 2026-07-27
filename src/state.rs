use rand::{seq::SliceRandom, Rng};

pub trait State {
    type Action: Copy;
    fn default_action() -> Self::Action;

    fn player_has_won(&self, player: usize) -> bool;
    fn is_terminal(&self) -> bool;
    fn get_legal_actions(&self) -> Vec<Self::Action>;

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

    fn to_play(&self) -> usize;
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

    fn reward(&self, to_play: usize) -> f32;
    fn render(&self);
}
