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

        this.socket.on(
            "ai_move",
            (data)=>{
                console.debug(`The AI decision is \n${data}`)
            }
        );
    }

    sendPlayerMove(environment){
        if(this.socket){
            this.socket.emit("player_action", environment);
        }
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