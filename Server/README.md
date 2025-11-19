# Server-Side Module

## Overview  
This module handles all server-side communication and hosts the AI model that interacts with the game environment. It processes incoming game data, performs reinforcement learning updates, and sends real-time actions back to the game through WebSockets.

### Key Features  
- **Data Processing:** Normalizes, structures, and prepares incoming state information before passing it to the AI agent.  
- **WebSocket Integration:** Provides real-time, two-way communication between the game engine and the AI model.  
- **Model Updates:** Supports dynamic learning and model updates for adaptive, improving agent behavior.  
- **Error Handling:** Includes logging, exception control, and fail-safe mechanisms to ensure stable operation.  

![Monitoring Screenshot](assets/screenshots/monitoring.png)

## Technologies  
This system uses **Flask** combined with **Flask-SocketIO** to handle real-time communication and AI inference.

### Dependencies  
Make sure you have the following installed:  
- Python 3.x  
- Flask  
- Flask-SocketIO  
- PyTorch  

## How to Run  

1. **Activate the virtual environment:**  
```bash
source server_env/bin/activate
```

2. **Install dependencies:**  
```bash
pip install -r requirements.txt
```

3. **Start the server:**  
```bash
python run.py
```

4. **Verify the connection:**  
Open the game or a WebSocket client and confirm that the server is sending and receiving events correctly.

---

# Mathematical Background

## Context  
The game `PongGame` sends the following JSON structure to the server:

```json
{
  "combo_smash": 0,
  "width": 800,
  "height": 600,
  "myself": {
    "score": 0,
    "position": { "x": 700, "y": 308.2 }
  },
  "player": {
    "score": 0,
    "position": { "x": 100, "y": 300 }
  },
  "ball": {
    "position": { "x": 406.5, "y": 306.5 }
  }
}
```

From this, we construct the state vector **s**, which currently includes the following 11 features:

```text
[
  combo_smash,
  my_score,
  my_x,
  my_y,
  player_score,
  player_x,
  player_y,
  ball_x,
  ball_y,
  relative_ball_position_y (ball.y - myself.y),
  relative_ball_position_x (ball.x - myself.x)
]
```

### Actions  
The agent can choose from three possible actions:

```
[
  up,
  down,
  stay
]
```

---

# Reward Function Fundamentals  

The reward function is designed around:  
- How close the racket is to the ball  
- Whether the agent hits the ball  
- Whether either player scores  
- Time-step penalties to encourage efficiency  

Mathematically, the reward components are:

```
[
  +1.0   if the agent scores
  -1.0   if the opponent scores
  +0.3   if the agent hits the ball
  +0.05  if the vertical distance decreases
  -0.05  if the vertical distance increases
  -0.001 time penalty per step
]
```

The total reward is defined as:

```
reward = reward_score 
       + reward_lose 
       + reward_hit 
       + reward_alignment 
       + reward_time
```