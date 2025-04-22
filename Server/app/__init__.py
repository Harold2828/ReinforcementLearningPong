from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ass'  # Remember to change this

# Initialize Flask-SocketIO and associate it with the app instance
socketio = SocketIO(app, cors_allowed_origins="*")
