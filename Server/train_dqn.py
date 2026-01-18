# Server/train_dqn.py
from __future__ import annotations

import os
import numpy as np

from app.ai.dqn_agent import DQNAgent
from app.ai.utils import json_to_state


class SimplePongEnv:
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.my_x = 700
        self.player_x = 100

        self.my_y = float(np.random.uniform(0, self.height))
        self.player_y = float(np.random.uniform(0, self.height))

        self.ball_x = float(np.random.uniform(0, self.width))
        self.ball_y = float(np.random.uniform(0, self.height))

        self.my_score = 0
        self.player_score = 0
        self.combo_smash = 0

        return self._get_state()

    def _to_json(self):
        return {
            "combo_smash": self.combo_smash,
            "myself": {
                "score": self.my_score,
                "position": {"x": float(self.my_x), "y": float(self.my_y)},
            },
            "player": {
                "score": self.player_score,
                "position": {"x": float(self.player_x), "y": float(self.player_y)},
            },
            "ball": {"position": {"x": float(self.ball_x), "y": float(self.ball_y)}},
        }

    def _get_state(self):
        return json_to_state(self._to_json())

    def step(self, action: int):
        # actions: 0=up, 1=down, 2=stay
        if action == 0:
            self.my_y += 15.0
        elif action == 1:
            self.my_y -= 15.0

        self.my_y = float(np.clip(self.my_y, 0.0, float(self.height)))

        # Move ball randomly
        self.ball_x += float(np.random.uniform(-25.0, 25.0))
        self.ball_y += float(np.random.uniform(-20.0, 20.0))
        self.ball_x = float(np.clip(self.ball_x, 0.0, float(self.width)))
        self.ball_y = float(np.clip(self.ball_y, 0.0, float(self.height)))

        # Reward: reduce vertical distance to ball (normalized)
        dy = abs(self.ball_y - self.my_y)
        reward = - (dy / float(self.height))

        done = False  # toy env: never ends
        return self._get_state(), float(reward), done


def train():
    print("Training DQN agent on SimplePongEnv")

    env = SimplePongEnv()

    # Reset once to get correct state_dim
    state = env.reset()
    state_dim = len(state)

    agent = DQNAgent(
        state_dim=state_dim,
        num_actions=3,
        lr=1e-3,
        batch_size=64,
    )

    episodes = 1000
    steps_per_episode = 300

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0

        for _ in range(steps_per_episode):
            # 1) choose action
            action = agent.select_action(state, explore=True)

            # 2) step env
            next_state, reward, done = env.step(action)

            # 3) store transition
            agent.remember(state, action, reward, next_state, done)

            # 4) train
            agent.train_step()

            # 5) advance
            state = next_state
            total_reward += reward

            if done:
                break

        print(f"Episode {episode} | Total reward: {total_reward:.2f}")

    # Save trained model
    os.makedirs("models", exist_ok=True)
    agent.save("models/dqn_pong.pt")
    print("Saved model to models/dqn_pong.pt")


if __name__ == "__main__":
    train()
