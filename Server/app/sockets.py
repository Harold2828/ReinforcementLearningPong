def register_sockets(socketio):
    @socketio.on('connect')
    def handle_connect():
        print("Client connected")

    @socketio.on('message')
    def handle_message(data):
        print("Message from client", data)