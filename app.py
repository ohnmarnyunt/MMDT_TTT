"""
app.py - Flask web server (Task 3)

The browser keeps the board and sends it to the server whenever it needs the AI.
The server is "stateless": it never remembers a game, so it works fine on Render
with several gunicorn workers.

Endpoints
  GET  /              -> the web page (templates/index.html)
  POST /api/move      -> {"board": [...9 cells...], "agent": "minimax"}  => AI's move + scores
  POST /api/simulate  -> {"agent_x": "random", "agent_o": "minimax", "games": 100} => stats
  GET  /health        -> "ok" (handy for checking the deployment)
"""

import os
import time

from flask import Flask, jsonify, render_template, request

from agent import AGENTS, MinimaxAgent, make_agent, simulate_match
from game import EMPTY, TicTacToe

app = Flask(__name__)

MAX_SIM_GAMES = 500


def _read_board(data):
    board = data.get("board")
    if not isinstance(board, list) or len(board) != 9:
        raise ValueError("board must be a list of 9 cells")
    return [c if c in ("X", "O") else EMPTY for c in board]


def _state(game):
    return {
        "board": game.board,
        "winner": game.check_winner(),
        "winning_line": game.get_winning_line(),
        "draw": game.is_draw(),
        "game_over": game.is_game_over(),
        "current_player": game.current_player,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return "ok"


@app.route("/api/move", methods=["POST"])
def api_move():
    data = request.get_json(silent=True) or {}
    try:
        game = TicTacToe(_read_board(data))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    agent_name = str(data.get("agent", "minimax")).lower()
    if agent_name not in AGENTS:
        return jsonify({"error": f"unknown agent '{agent_name}'"}), 400
    if game.is_game_over():
        return jsonify({"error": "game is already over", **_state(game)}), 400

    agent = make_agent(agent_name)
    player = game.current_player
    start = time.perf_counter()
    move = agent.get_move(game)
    elapsed_ms = (time.perf_counter() - start) * 1000

    info = {"agent": agent_name, "player": player, "move": move,
            "time_ms": round(elapsed_ms, 2)}

    if isinstance(agent, MinimaxAgent):
        info["scores"] = {str(k): v for k, v in agent.move_scores.items()}
        info["nodes_visited"] = agent.nodes_visited
        info["prune_count"] = agent.prune_count
        if data.get("compare"):
            # Same search WITHOUT pruning, to show how many nodes Alpha-Beta saved.
            plain = MinimaxAgent(use_pruning=False)
            plain_move = plain.get_move(game)
            info["nodes_without_pruning"] = plain.nodes_visited
            info["same_decision"] = (plain_move == move and plain.move_scores == agent.move_scores)
    elif agent_name == "heuristic":
        info["reason"] = agent.last_reason

    game.make_move(move, player)
    info.update(_state(game))
    return jsonify(info)


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    data = request.get_json(silent=True) or {}
    x_name = str(data.get("agent_x", "random")).lower()
    o_name = str(data.get("agent_o", "minimax")).lower()
    if x_name not in AGENTS or o_name not in AGENTS:
        return jsonify({"error": "unknown agent"}), 400
    try:
        n = max(1, min(int(data.get("games", 100)), MAX_SIM_GAMES))
    except (TypeError, ValueError):
        n = 100
    start = time.perf_counter()
    results = simulate_match(make_agent(x_name), make_agent(o_name), n)
    results.update({"agent_x": x_name, "agent_o": o_name,
                    "time_ms": round((time.perf_counter() - start) * 1000, 1)})
    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
