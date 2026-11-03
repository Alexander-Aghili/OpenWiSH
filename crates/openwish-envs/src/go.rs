//! 9x9 Go with the rules of the authors' Experiment 6 (pgx "go_9x9"): simple ko and suicide are illegal
//! (and masked); a move that repeats an earlier whole-board position ends the game and its player loses;
//! the game ends after two consecutive passes or 162 moves in total; area scoring, komi 7.5.
use std::collections::HashSet;

pub const N: usize = 9;
pub const P: usize = N * N; // 81 points
pub const PASS: usize = P; // move 81
/// The network input: three planes over the 81 points (`Game::encode`).
pub const INPUTS: usize = 3 * P;
pub const MAX_MOVES: u32 = 162;
pub const KOMI: f32 = 7.5;

/// Hash of a whole-board position (FNV-1a over the 81 points).
pub fn hash(b: &[i8; P]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for v in b {
        h ^= (*v as u8) as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum End {
    TwoPasses,
    MoveLimit,
    /// the player who repeated a position loses
    Repetition,
}

#[derive(Clone, Debug)]
pub struct Game {
    /// 1 = black, -1 = white, 0 = empty
    pub board: [i8; P],
    /// 1 = black to move, -1 = white
    pub to_move: i8,
    pub ko: Option<usize>,
    pub passes_in_row: u32,
    pub moves: u32,
    pub history: HashSet<u64>,
    pub end: Option<End>,
    /// +1 black won, -1 white won (set when the game ends)
    pub winner: i8,
    /// stones captured by black / by white
    pub captured: [u32; 2],
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

impl Game {
    pub fn new() -> Game {
        let board = [0i8; P];
        let mut history = HashSet::new();
        history.insert(hash(&board));
        Game {
            board,
            to_move: 1,
            ko: None,
            passes_in_row: 0,
            moves: 0,
            history,
            end: None,
            winner: 0,
            captured: [0, 0],
        }
    }
    /// Build a position from 9 strings (X black, O white, . empty). History holds only this position.
    pub fn from_strings(rows: &[&str], to_move: i8) -> Game {
        let mut g = Game::new();
        for (r, row) in rows.iter().enumerate().take(N) {
            for (c, ch) in row.chars().enumerate().take(N) {
                g.board[r * N + c] = match ch {
                    'X' | 'x' | 'B' | 'b' => 1,
                    'O' | 'o' | 'W' | 'w' => -1,
                    _ => 0,
                };
            }
        }
        g.to_move = to_move;
        g.history.clear();
        g.history.insert(hash(&g.board));
        g
    }
    pub fn neighbours(p: usize) -> impl Iterator<Item = usize> {
        let (r, c) = (p / N, p % N);
        let mut v = [usize::MAX; 4];
        if r > 0 {
            v[0] = p - N;
        }
        if r + 1 < N {
            v[1] = p + N;
        }
        if c > 0 {
            v[2] = p - 1;
        }
        if c + 1 < N {
            v[3] = p + 1;
        }
        v.into_iter().filter(|x| *x != usize::MAX)
    }
    /// (stones of the group at p, its liberty count)
    pub fn group(&self, p: usize) -> (Vec<usize>, usize) {
        let colour = self.board[p];
        let mut seen = [false; P];
        let mut libs = [false; P];
        let mut stack = vec![p];
        let mut stones = vec![];
        seen[p] = true;
        while let Some(q) = stack.pop() {
            stones.push(q);
            for n in Self::neighbours(q) {
                if self.board[n] == 0 {
                    libs[n] = true;
                } else if self.board[n] == colour && !seen[n] {
                    seen[n] = true;
                    stack.push(n);
                }
            }
        }
        (stones, libs.iter().filter(|x| **x).count())
    }
    /// Legal moves (82 entries, the last is pass): empty, not suicide, not the ko point.
    pub fn legal(&self) -> [bool; P + 1] {
        let mut out = [false; P + 1];
        out[PASS] = true;
        if self.end.is_some() {
            return out;
        }
        for (p, legal) in out[..P].iter_mut().enumerate() {
            *legal = self.board[p] == 0 && self.ko != Some(p) && !self.is_suicide(p);
        }
        out
    }
    /// Whether a stone of the player to move on the empty point `p` would have no liberty: it has no empty
    /// neighbour, joins no friendly group with another liberty, and takes the last liberty of no opponent group.
    fn is_suicide(&self, p: usize) -> bool {
        !Self::neighbours(p).any(|n| {
            let v = self.board[n];
            if v == 0 {
                return true;
            }
            let (_, libs) = self.group(n);
            (v == self.to_move && libs > 1) || (v == -self.to_move && libs == 1)
        })
    }
    /// Play a legal move (panics on an illegal one). Returns the number of stones captured.
    pub fn play(&mut self, m: usize) -> u32 {
        assert!(self.end.is_none(), "the game is over");
        let me = self.to_move;
        let mut taken = 0u32;
        if m == PASS {
            self.passes_in_row += 1;
            self.ko = None;
        } else {
            assert!(self.legal()[m], "illegal move {m}");
            self.board[m] = me;
            let mut last_taken = None;
            for n in Self::neighbours(m) {
                if self.board[n] == -me {
                    let (stones, libs) = self.group(n);
                    if libs == 0 {
                        for s in &stones {
                            self.board[*s] = 0;
                        }
                        taken += stones.len() as u32;
                        last_taken = stones.first().copied();
                    }
                }
            }
            let (mine, libs) = self.group(m);
            self.ko = if taken == 1 && mine.len() == 1 && libs == 1 {
                last_taken
            } else {
                None
            };
            self.captured[if me == 1 { 0 } else { 1 }] += taken;
            self.passes_in_row = 0;
            if !self.history.insert(hash(&self.board)) {
                self.moves += 1;
                self.end = Some(End::Repetition);
                self.winner = -me;
                self.to_move = -me;
                return taken;
            }
        }
        self.moves += 1;
        self.to_move = -me;
        if self.passes_in_row >= 2 {
            self.finish(End::TwoPasses);
        } else if self.moves >= MAX_MOVES {
            self.finish(End::MoveLimit);
        }
        taken
    }

    fn finish(&mut self, e: End) {
        let (b, w) = self.area();
        self.end = Some(e);
        self.winner = if b > w { 1 } else { -1 };
    }
    /// Area score: (black, white incl. komi)
    pub fn area(&self) -> (f32, f32) {
        let mut score = [0f32; 2];
        let mut seen = [false; P];
        for p in 0..P {
            match self.board[p] {
                1 => score[0] += 1.0,
                -1 => score[1] += 1.0,
                _ if !seen[p] => {
                    // flood the empty region and see which colours border it
                    let mut stack = vec![p];
                    seen[p] = true;
                    let (mut size, mut b, mut w) = (0, false, false);
                    while let Some(q) = stack.pop() {
                        size += 1;
                        for n in Self::neighbours(q) {
                            match self.board[n] {
                                1 => b = true,
                                -1 => w = true,
                                _ if !seen[n] => {
                                    seen[n] = true;
                                    stack.push(n);
                                }
                                _ => {}
                            }
                        }
                    }
                    if b && !w {
                        score[0] += size as f32;
                    } else if w && !b {
                        score[1] += size as f32;
                    }
                }
                _ => {}
            }
        }
        (score[0], score[1] + KOMI)
    }
    /// The 243-number input from the point of view of the player to move.
    pub fn encode(&self, x: &mut [f32]) {
        let white = if self.to_move == -1 { 1.0 } else { 0.0 };
        for p in 0..P {
            let v = self.board[p] * self.to_move;
            x[p * 3] = if v == 1 { 1.0 } else { 0.0 };
            x[p * 3 + 1] = if v == -1 { 1.0 } else { 0.0 };
            x[p * 3 + 2] = white;
        }
    }
    pub fn rows(&self) -> Vec<String> {
        (0..N)
            .map(|r| {
                (0..N)
                    .map(|c| match self.board[r * N + c] {
                        1 => 'X',
                        -1 => 'O',
                        _ => '.',
                    })
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(r: usize, c: usize) -> usize {
        r * N + c
    }
    const E: &str = ".........";

    #[test]
    fn capture_single_and_group() {
        let mut g = Game::from_strings(&["XO.......", E, E, E, E, E, E, E, E], 1);
        assert_eq!(g.play(at(1, 1)), 0);
        g.play(PASS);
        assert_eq!(g.play(at(0, 2)), 1);
        assert_eq!(g.board[at(0, 1)], 0);
        assert_eq!(g.captured, [1, 0]);
        let mut g = Game::from_strings(&[".XO......", "XXO......", E, E, E, E, E, E, E], 1);
        g.play(at(0, 3));
        g.play(PASS);
        g.play(at(1, 3));
        g.play(PASS);
        assert_eq!(g.play(at(2, 2)), 2);
        assert_eq!((g.board[at(0, 2)], g.board[at(1, 2)]), (0, 0));
    }

    #[test]
    fn suicide_is_illegal_but_capture_is_not() {
        let g = Game::from_strings(&[".X.......", "X........", E, E, E, E, E, E, E], -1);
        assert!(!g.legal()[at(0, 0)], "white at (0,0) would have no liberties");
        // here white at (0,0) takes the last liberty of both black stones, so it captures and is legal
        let g = Game::from_strings(&[".XO......", "XO.......", "O........", E, E, E, E, E, E], -1);
        assert!(g.legal()[at(0, 0)]);
        assert!(g.legal()[PASS]);
        assert!(!g.legal()[at(0, 1)], "occupied");
    }

    #[test]
    fn simple_ko() {
        let mut g = Game::from_strings(&[".XO......", "XO.O.....", ".XO......", E, E, E, E, E, E], 1);
        assert_eq!(g.play(at(1, 2)), 1);
        assert_eq!(g.ko, Some(at(1, 1)));
        assert!(!g.legal()[at(1, 1)]);
        g.play(at(8, 8));
        g.play(at(8, 0));
        assert!(g.legal()[at(1, 1)], "ko can be retaken after a move elsewhere");
    }

    #[test]
    fn two_passes_end_with_area_winner() {
        let mut g = Game::new();
        g.play(at(4, 4));
        g.play(PASS);
        g.play(PASS);
        assert_eq!(g.end, Some(End::TwoPasses));
        assert_eq!(g.winner, 1, "81 against 7.5");
        let mut g = Game::new();
        g.play(at(0, 0));
        g.play(at(0, 1));
        g.play(at(1, 1));
        assert_eq!(g.play(at(1, 0)), 1, "white captures (0,0)");
        g.play(PASS);
        assert_eq!(g.end, None);
        g.play(PASS);
        assert_eq!(g.end, Some(End::TwoPasses));
    }

    #[test]
    fn positional_superko_mover_loses() {
        let mut k = Game::new();
        let mut target = k.board;
        target[at(0, 0)] = 1;
        k.history.insert(hash(&target));
        k.play(at(0, 0));
        assert_eq!(k.end, Some(End::Repetition));
        assert_eq!(k.winner, -1, "black repeated a position, so black loses");
    }

    #[test]
    fn move_limit_and_area_scoring() {
        let mut g = Game::new();
        while g.end.is_none() {
            let m = if g.moves.is_multiple_of(2) {
                (g.moves as usize / 2 * 7) % P
            } else {
                PASS
            };
            let legal = g.legal();
            g.play(if legal[m] {
                m
            } else {
                legal.iter().position(|x| *x).unwrap()
            });
        }
        assert!(g.moves <= MAX_MOVES);
        let col = ["....X.O.."; 9];
        let g = Game::from_strings(&col, 1);
        assert_eq!(g.area(), (45.0, 27.0 + KOMI));
    }

    #[test]
    fn encoding_is_from_the_mover_point_of_view() {
        let mut g = Game::new();
        g.play(at(0, 1));
        let mut x = [0f32; INPUTS];
        g.encode(&mut x);
        assert_eq!(x[at(0, 1) * 3 + 1], 1.0, "black is the opponent of white");
        assert_eq!(x[at(0, 1) * 3], 0.0);
        assert_eq!(x[2], 1.0, "plane 2 = white to move");
        assert_eq!(x.iter().sum::<f32>(), 1.0 + 81.0);
    }
}
