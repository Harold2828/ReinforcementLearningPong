# app/ai/utils.py
from __future__ import annotations
from typing import Dict
import numpy as np


def json_to_state(game_state: Dict) -> np.ndarray:
    """
    Convert the game JSON into a flat numeric state vector.

    Expected JSON format:
    {
        "combo_smash": 0,
        "width": 800,
        "height": 600,
        "myself": {
            "score": 0,
            "position": {
                "x": 700,
                "y": 308.3333333333333
            }
        },
        "player": {
            "score": 0,
            "position": {
                "x": 100,
                "y": 300
            }
        },
        "ball": {
            "position": {
                "x": 406.66666666666663,
                "y": 306.66666666666663
            }
        }
    }
    """

    width = game_state.get("width", 800)
    height = game_state.get("height", 600)

    combo_smash = float(game_state.get("combo_smash", 0.0))

    myself = game_state["myself"]
    my_score = myself["score"]
    my_x = myself["position"]["x"] / width
    my_y = myself["position"]["y"] / height

    player = game_state["player"]
    player_score = player["score"]
    player_x = player["position"]["x"] / width
    player_y = player["position"]["y"] / height

    ball = game_state["ball"]
    ball_x = ball["position"]["x"] / width
    ball_y = ball["position"]["y"] / height

    # relative features
    dy_me_ball = ball_y - my_y
    dy_player_ball = ball_y - player_y
    dx_ball_me = ball_x - my_x

    state = np.array(
        [
            my_score,
            player_score,
            my_y,
            player_y,
            ball_x,
            ball_y,
            dy_me_ball,
            dy_player_ball,
            dx_ball_me,
            combo_smash,
        ],
        dtype=np.float32
    )

    return state
