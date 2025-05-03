from flask_socketio import emit
import random

def register_sockets(socketio):
    @socketio.on('connect')
    def on_connect():
        print("✅ Client connected")

    @socketio.on('player_action')
    def on_player_action(data):
        print(f"📥 Player sent: {data}")
        
        # Simulate AI response — later replaced by PyTorch
        ai_direction = random.choice(['left', 'right', 'stay'])

        # Send it back to client
        emit('ai_move', {'direction': ai_direction})
