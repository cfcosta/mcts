use crate::games::{bit_count, nth_set_bit, random_below};
use crate::state::State;

#[derive(Debug, Clone, Copy)]
pub struct TicTacToe {
    board_x: u16,
    board_o: u16,
    // Current player: 0 => X, 1 => O
    current_player: u8,
}

const WIN_PATTERNS: [u16; 8] = [
    // rows
    0b111000000,
    0b000111000,
    0b000000111,
    // cols
    0b100100100,
    0b010010010,
    0b001001001,
    // diags
    0b100010001,
    0b001010100,
];

const WIN_TABLE: [bool; 512] = {
    let mut table = [false; 512];
    let mut board = 0;
    while board < table.len() {
        let mut pattern = 0;
        while pattern < WIN_PATTERNS.len() {
            if (board as u16 & WIN_PATTERNS[pattern]) == WIN_PATTERNS[pattern] {
                table[board] = true;
                break;
            }
            pattern += 1;
        }
        board += 1;
    }
    table
};

impl State for TicTacToe {
    type Action = (u8, u8);

    fn default_action() -> Self::Action {
        (0, 0)
    }

    fn player_has_won(&self, player: usize) -> bool {
        let board = match player {
            0_usize => self.board_x,
            _ => self.board_o,
        };
        WIN_TABLE[board as usize]
    }

    fn is_terminal(&self) -> bool {
        self.player_has_won(0)
            || self.player_has_won(1)
            || board_is_filled(self.board_x, self.board_o)
    }

    fn get_legal_actions(&self) -> Vec<Self::Action> {
        determine_legal_actions(self.board_x, self.board_o)
    }

    fn get_random_legal_action<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Self::Action {
        let empty = !(self.board_x | self.board_o) & 0x1FF;
        let rank = random_below(rng, bit_count(empty));
        let pos = nth_set_bit(empty, rank);
        (pos / 3, pos % 3)
    }

    fn to_play(&self) -> usize {
        self.current_player as usize
    }

    fn step(&self, action: Self::Action) -> Self {
        let mut next = *self;
        next.step_in_place(action);
        next
    }

    fn step_in_place(&mut self, action: Self::Action) {
        if self.current_player == 0 {
            set_bit(&mut self.board_x, action.0 * 3 + action.1);
        } else {
            set_bit(&mut self.board_o, action.0 * 3 + action.1);
        }
        self.current_player = 1 - self.current_player;
    }

    fn reward(&self, to_play: usize) -> f32 {
        if self.player_has_won(to_play) {
            -1.0
        } else if self.player_has_won(1 - to_play) {
            1.0
        } else {
            0.0
        }
    }

    fn render(&self) {
        println!("X: player 0, O: player 1\n");
        for i in (0..3).rev() {
            let mut current_line: Vec<String> = Vec::with_capacity(3);
            for j in 0..3 {
                let pos = i * 3 + j;

                let mask = 1 << pos;
                if (self.board_x & mask) != 0 {
                    current_line.push("X".to_string());
                } else if (self.board_o & mask) != 0 {
                    current_line.push("O".to_string());
                } else {
                    current_line.push(" ".to_string());
                }
            }
            println!(
                " {} | {} | {}",
                current_line[0], current_line[1], current_line[2]
            );
            if i > 0 {
                println!("---------");
            }
        }
        println!();
    }
}

impl TicTacToe {
    pub fn new() -> TicTacToe {
        TicTacToe {
            board_x: 0,
            board_o: 0,
            current_player: 0,
        }
    }
}

impl Default for TicTacToe {
    fn default() -> Self {
        Self::new()
    }
}

fn determine_legal_actions(board_x: u16, board_o: u16) -> Vec<(u8, u8)> {
    let mut actions: Vec<(u8, u8)> = Vec::with_capacity(9);
    for pos in 0..9 {
        let mask = 1 << pos;
        if (board_x & mask) == 0 && (board_o & mask) == 0 {
            actions.push((pos / 3, pos % 3));
        }
    }
    actions
}

fn board_is_filled(board_x: u16, board_o: u16) -> bool {
    (board_x | board_o) == 0x1FF
}

fn set_bit(board: &mut u16, pos: u8) {
    *board |= 1 << pos;
}
