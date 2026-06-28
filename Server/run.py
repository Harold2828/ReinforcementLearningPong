import os

from app import create_app, socketio

app = create_app()

if __name__ == '__main__':
    debugEnabled = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    socketio.run(app, debug=debugEnabled, host='0.0.0.0', port=5001)
