"""
agent.py - AI agents for Tic-Tac-Toe (Task 2)

Three "brains" that all answer the same question:  "Given this board, which cell do I play?"

  RandomAgent    (Easy)        - picks any empty cell at random.
  HeuristicAgent (Medium)      - follows simple rules: Win > Block > Center > Corner > Edge.
  MinimaxAgent   (Unbeatable)  - looks ahead at every possible future game (Minimax),
                                 skipping branches that cannot matter (Alpha-Beta pruning).

Every agent has the method  get_move(game) -> int  (a cell index 0..8).
The agent plays for whoever's turn it is (game.current_player).
"""

import math
import random

from game import TicTacToe

CORNERS = [0, 2, 6, 8]
EDGES = [1, 3, 5, 7]
CENTER = 4


class RandomAgent:
    name = "Random"

    def __init__(self, player=None, seed=None):
        self.player = player
        self.rng = random.Random(seed)

    def get_move(self, game):
        return self.rng.choice(game.get_available_moves())


class HeuristicAgent:
    name = "Heuristic"

    def __init__(self, player=None, seed=None):
        self.player = player
        self.rng = random.Random(seed)
        self.last_reason = ""

    def _winning_move_for(self, game, player):
        """A cell that would complete three-in-a-row for `player`, else None."""
        for pos in game.get_available_moves():
            game.make_move(pos, player)
            won = game.check_winner() == player
            game.undo_move(pos)
            if won:
                return pos
        return None

    def get_move(self, game):
        me = game.current_player
        opponent = TicTacToe.other(me)
        moves = game.get_available_moves()

        # Rule 1: if I can win right now, win.
        pos = self._winning_move_for(game, me)
        if pos is not None:
            self.last_reason = "win"
            return pos
        # Rule 2: if the opponent could win next move, block that cell.
        pos = self._winning_move_for(game, opponent)
        if pos is not None:
            self.last_reason = "block"
            return pos
        # Rule 3: take the center.
        if CENTER in moves:
            self.last_reason = "center"
            return CENTER
        # Rule 4: take a corner.  Rule 5: take an edge.
        for reason, group in (("corner", CORNERS), ("edge", EDGES)):
            free = [p for p in group if p in moves]
            if free:
                self.last_reason = reason
                return self.rng.choice(free)
        return moves[0]


class MinimaxAgent:
    name = "Minimax"

    WIN_SCORE = 10
    _cache = {}   # board -> finished search, shared by all Minimax agents (speeds up benchmarks)

    def __init__(self, player=None, use_pruning=True):
        self.player = player
        self.use_pruning = use_pruning
        # Search statistics, refreshed on every get_move call (shown in the visualizer).
        self.nodes_visited = 0
        self.prune_count = 0
        self.move_scores = {}

    # ----- scoring a finished game -----
    def _evaluate(self, game, me, depth):
        winner = game.check_winner()
        if winner == me:
            return self.WIN_SCORE - depth      # win sooner  -> bigger score
        if winner is not None:
            return depth - self.WIN_SCORE      # lose later  -> less negative score
        return 0                               # draw

    # ----- the recursive search -----
    def minimax(self, game, depth, is_maximizing, me, alpha=-math.inf, beta=math.inf):
        self.nodes_visited += 1

        if game.is_game_over():
            return self._evaluate(game, me, depth)

        opponent = TicTacToe.other(me)

        if is_maximizing:                       # my turn: pick the highest score
            best = -math.inf
            for pos in game.get_available_moves():
                game.make_move(pos, me)
                score = self.minimax(game, depth + 1, False, me, alpha, beta)
                game.undo_move(pos)
                best = max(best, score)
                if self.use_pruning:
                    alpha = max(alpha, best)
                    if alpha >= beta:           # opponent will never allow this branch
                        self.prune_count += 1
                        break
            return best
        else:                                   # opponent's turn: assume they pick the lowest
            best = math.inf
            for pos in game.get_available_moves():
                game.make_move(pos, opponent)
                score = self.minimax(game, depth + 1, True, me, alpha, beta)
                game.undo_move(pos)
                best = min(best, score)
                if self.use_pruning:
                    beta = min(beta, best)
                    if alpha >= beta:
                        self.prune_count += 1
                        break
            return best

    def get_move(self, game):
        me = game.current_player
        key = (tuple(game.board), self.use_pruning)
        if key in MinimaxAgent._cache:           # same board searched before -> reuse result
            move, self.move_scores, self.nodes_visited, self.prune_count = MinimaxAgent._cache[key]
            self.move_scores = dict(self.move_scores)
            return move

        work = game.copy()                      # never touch the real game board
        self.nodes_visited = 0
        self.prune_count = 0
        self.move_scores = {}

        # Score every candidate move separately so the visualizer can show
        # an exact score for each empty cell.
        for pos in work.get_available_moves():
            work.make_move(pos, me)
            self.move_scores[pos] = self.minimax(work, 1, False, me)
            work.undo_move(pos)

        best_score = max(self.move_scores.values())
        # Tie-break: first cell with the best score (deterministic).
        move = min(p for p, s in self.move_scores.items() if s == best_score)
        MinimaxAgent._cache[key] = (move, dict(self.move_scores), self.nodes_visited, self.prune_count)
        return move


AGENTS = {
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
    "minimax": MinimaxAgent,
}


def make_agent(name):
    return AGENTS[name.lower()]()


def play_game(agent_x, agent_o):
    """Play one full game. Returns 'X', 'O', or None for a draw."""
    game = TicTacToe()
    while not game.is_game_over():
        agent = agent_x if game.current_player == 'X' else agent_o
        game.make_move(agent.get_move(game), game.current_player)
    return game.check_winner()


def simulate_match(agent_x, agent_o, n_games=100):
    """Play n_games between two agents (agent_x always starts) and count results."""
    results = {"games": n_games, "x_wins": 0, "o_wins": 0, "draws": 0}
    for _ in range(n_games):
        winner = play_game(agent_x, agent_o)
        if winner == 'X':
            results["x_wins"] += 1
        elif winner == 'O':
            results["o_wins"] += 1
        else:
            results["draws"] += 1
    for key in ("x_wins", "o_wins", "draws"):
        results[key.replace("wins", "win_rate").replace("draws", "draw_rate")] = round(
            100 * results[key] / n_games, 1) if n_games else 0.0
    return results
