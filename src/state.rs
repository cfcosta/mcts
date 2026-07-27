use rand::seq::SliceRandom;

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
    fn get_random_legal_action(&self) -> Self::Action {
        *self
            .get_legal_actions()
            .choose(&mut rand::thread_rng())
            .expect("no legal actions in a non-terminal state")
    }

    fn to_play(&self) -> usize;
    fn step(&self, action: Self::Action) -> Self;
    fn reward(&self, to_play: usize) -> f32;
    fn render(&self);
}
