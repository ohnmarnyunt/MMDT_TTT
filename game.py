"""
game.py - Tic-Tac-Toe game engine (Task 1)

The board is a list of 9 cells, numbered like this:

     0 | 1 | 2
    ---+---+---
     3 | 4 | 5
    ---+---+---
     6 | 7 | 8

Each cell holds 'X', 'O', or ' ' (a space = empty).
"""

EMPTY = ' '

# All 8 ways to get three in a row: 3 rows, 3 columns, 2 diagonals.
WINNING_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
]


class TicTacToe:
    def __init__(self, board=None):
        # Start from an empty board, or copy a board we were given.
        self.board = list(board) if board is not None else [EMPTY] * 9
        if len(self.board) != 9:
            raise ValueError("Board must have exactly 9 cells")

    # ---------- basic helpers ----------
    @property
    def current_player(self):
        """X always moves first, so whose turn it is follows from counting the marks."""
        return 'X' if self.board.count('X') == self.board.count('O') else 'O'

    def reset(self):
        self.board = [EMPTY] * 9

    def copy(self):
        return TicTacToe(self.board)

    @staticmethod
    def other(player):
        return 'O' if player == 'X' else 'X'

    # ---------- moves ----------
    def get_available_moves(self):
        """List of empty cell indexes, e.g. [0, 2, 5]."""
        return [i for i, cell in enumerate(self.board) if cell == EMPTY]

    def make_move(self, pos, player=None):
        """Put `player` on cell `pos`. Returns True if legal, False otherwise."""
        player = player or self.current_player
        if not isinstance(pos, int) or not 0 <= pos <= 8:
            return False
        if self.board[pos] != EMPTY or self.is_game_over():
            return False
        self.board[pos] = player
        return True

    def undo_move(self, pos):
        """Take a move back (used by Minimax to 'try' a move and then backtrack)."""
        if 0 <= pos <= 8 and self.board[pos] != EMPTY:
            self.board[pos] = EMPTY
            return True
        return False

    # ---------- results ----------
    def get_winning_line(self):
        """The 3 cells that make a win, e.g. (0, 4, 8), or None."""
        for a, b, c in WINNING_LINES:
            if self.board[a] != EMPTY and self.board[a] == self.board[b] == self.board[c]:
                return (a, b, c)
        return None

    def check_winner(self):
        """'X' or 'O' if someone has three in a row, otherwise None."""
        line = self.get_winning_line()
        return self.board[line[0]] if line else None

    def is_draw(self):
        """Board full and nobody won."""
        return EMPTY not in self.board and self.check_winner() is None

    def is_game_over(self):
        return self.check_winner() is not None or EMPTY not in self.board

    def __str__(self):
        rows = [" | ".join(self.board[r * 3:r * 3 + 3]) for r in range(3)]
        return "\n---------\n".join(rows)
