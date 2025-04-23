from flask import Flask
from flask_socketio import SocketIO

socketio = SocketIO(cors_allowed_origins="*")

def create_app():
    app = Flask(__name__)
    socketio.init_app(app)
    from .sockets import register_sockets
    register_sockets(socketio)
    return app
