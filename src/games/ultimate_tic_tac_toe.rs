use crate::state::State;
use rand::Rng;

#[derive(Debug, Clone, Copy)]
pub struct UltimateTicTacToe {
    board_x: [u16; 9],
    board_o: [u16; 9],
    macro_board_x: u16,
    macro_board_o: u16,
    current_player: u8,
    // The mini-board selected by the previous move, or 9 for the initial state.
    next_board: u8,
    occupied: u8,
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

impl State for UltimateTicTacToe {
    type Action = (u8, u8, u8); // Mini-board, Row, Col

    fn default_action() -> Self::Action {
        (0, 0, 0)
    }

    fn player_has_won(&self, player: usize) -> bool {
        let board = match player {
            0_usize => self.macro_board_x,
            _ => self.macro_board_o,
        };
        WIN_TABLE[board as usize]
    }

    fn is_terminal(&self) -> bool {
        self.occupied == 81 || self.player_has_won(0) || self.player_has_won(1)
    }

    fn get_legal_actions(&self) -> Vec<Self::Action> {
        determine_legal_actions(
            &self.board_x,
            &self.board_o,
            self.macro_board_x,
            self.macro_board_o,
            self.next_board,
        )
    }

    fn get_random_legal_action(&self) -> Self::Action {
        if self.next_board < 9 {
            let board = self.next_board as usize;
            let board_mask = 1 << board;
            let empty = !(self.board_x[board] | self.board_o[board]) & 0x1FF;
            if ((self.macro_board_x | self.macro_board_o) & board_mask) == 0 && empty != 0 {
                let target = rand::thread_rng().gen_range(0..empty.count_ones());
                let pos = nth_set_bit(empty, target);
                return (board as u8, pos / 3, pos % 3);
            }
        }

        let mut empty_count: u32 = self
            .board_x
            .iter()
            .zip(&self.board_o)
            .map(|(&x, &o)| (!(x | o) & 0x1FF).count_ones())
            .sum();
        let mut target = rand::thread_rng().gen_range(0..empty_count);
        for board in 0..9 {
            let empty = !(self.board_x[board] | self.board_o[board]) & 0x1FF;
            empty_count = empty.count_ones();
            if target < empty_count {
                let pos = nth_set_bit(empty, target);
                return (board as u8, pos / 3, pos % 3);
            }
            target -= empty_count;
        }
        unreachable!("non-terminal Ultimate Tic-Tac-Toe state has no legal action")
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
        let mini_board = action.0 as usize;
        let position = action.1 * 3 + action.2;
        let was_empty = ((self.board_x[mini_board] | self.board_o[mini_board])
            & (1 << position))
            == 0;
        if self.current_player == 0 {
            set_bit(&mut self.board_x, mini_board, position);
        } else {
            set_bit(&mut self.board_o, mini_board, position);
        }
        update_macro_board(
            &self.board_x,
            &self.board_o,
            &mut self.macro_board_x,
            &mut self.macro_board_o,
            mini_board,
        );
        self.current_player = 1 - self.current_player;
        self.next_board = position;
        self.occupied += u8::from(was_empty);
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
        for big_row in 0..3 {
            for sub_row in 0..3 {
                let mut row_segments: Vec<String> = Vec::with_capacity(3);
                for big_col in 0..3 {
                    let mini_board_index = big_row * 3 + big_col;
                    let mut segment = String::new();
                    for sub_col in 0..3 {
                        let pos = sub_row * 3 + sub_col;
                        let mask = 1 << pos;

                        if (self.board_x[mini_board_index] & mask) != 0 {
                            segment.push('X');
                        } else if (self.board_o[mini_board_index] & mask) != 0 {
                            segment.push('O');
                        } else {
                            segment.push(' ');
                        }
                        if sub_col < 2 {
                            segment.push('|');
                        }
                    }
                    row_segments.push(segment);
                }
                println!(
                    " {} || {} || {}",
                    row_segments[0], row_segments[1], row_segments[2]
                );
            }
            if big_row < 2 {
                println!("=======||=======||=======");
            }
        }
        println!();
    }
}

fn update_macro_board(
    board_x: &[u16; 9],
    board_o: &[u16; 9],
    macro_board_x: &mut u16,
    macro_board_o: &mut u16,
    board_to_check: usize,
) {
    let mini_board_x = board_x[board_to_check];
    let mini_board_o = board_o[board_to_check];

    if WIN_TABLE[mini_board_x as usize] {
        *macro_board_x |= 1 << board_to_check;
    } else if WIN_TABLE[mini_board_o as usize] {
        *macro_board_o |= 1 << board_to_check;
    }
}

fn determine_legal_actions(
    board_x: &[u16; 9],
    board_o: &[u16; 9],
    macro_board_x: u16,
    macro_board_o: u16,
    next_board: u8,
) -> Vec<(u8, u8, u8)> {
    if next_board < 9 {
        let next_board = next_board as usize;
        let board_mask = 1 << next_board;
        if ((macro_board_x | macro_board_o) & board_mask) == 0
            && (board_x[next_board] | board_o[next_board]) != 0x1FF
        {
            let mut actions = Vec::with_capacity(9);
            let mut empty = !(board_x[next_board] | board_o[next_board]) & 0x1FF;
            while empty != 0 {
                let pos = empty.trailing_zeros() as u8;
                actions.push((next_board as u8, pos / 3, pos % 3));
                empty &= empty - 1;
            }
            return actions;
        }
    }

    let mut actions = Vec::with_capacity(81);
    for i in 0..9 {
        let mut empty = !(board_x[i] | board_o[i]) & 0x1FF;
        while empty != 0 {
            let pos = empty.trailing_zeros() as u8;
            actions.push((i as u8, pos / 3, pos % 3));
            empty &= empty - 1;
        }
    }
    actions
}

#[inline]
fn nth_set_bit(mut bits: u16, mut n: u32) -> u8 {
    while n != 0 {
        bits &= bits - 1;
        n -= 1;
    }
    bits.trailing_zeros() as u8
}

fn set_bit(board: &mut [u16; 9], mini_board: usize, pos: u8) {
    board[mini_board] |= 1 << pos;
}

impl UltimateTicTacToe {
    pub fn new() -> UltimateTicTacToe {
        UltimateTicTacToe {
            board_x: [0; 9],
            board_o: [0; 9],
            macro_board_x: 0,
            macro_board_o: 0,
            current_player: 0,
            next_board: 9,
            occupied: 0,
        }
    }
}

impl Default for UltimateTicTacToe {
    fn default() -> Self {
        Self::new()
    }
}
