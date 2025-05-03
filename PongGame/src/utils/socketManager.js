//import io from "socket.io-client";

class SocketManager{

    constructor(serverUrl){

        this.socket = io(serverUrl);

        this.socket.on(
            "connect",
            (data)=>{

                console.debug("Connected to server");
            }
        );


    }

    sendPlayerMove(environment){
        if(this.socket){
            this.socket.emit("player_action", environment);
        }
    }

    getMachineMove(){
        return new Promise((resolve, reject) => {
            if (this.socket) {
                this.socket.on("ai_move", (data) => {
                    resolve(data);
                });
            } else {
                reject(new Error("Socket is not connected"));
            }
        });
    }

    on(event, callback){
        if(this.socket){
            this.socket.on(event, callback);
        }
    }

    disconnect(){
        if(this.socket){
            this.socket.disconnect();
        }
    }

}

export default SocketManager;