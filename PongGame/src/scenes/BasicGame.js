import Phaser from 'phaser';
import SocketManager from '../utils/socketManager';
import { randomNormal } from '../utils/customRandom';

class BasicGame extends Phaser.Scene {

    constructor() {
    
        super({ key: 'BasicGame' });

        //Mechanics
        this.background = {
            court:{
                image: null
            },
            ball:{
                image: null,
                sound:null,
                initial:{
                    position:{
                        x: 400,
                        y: 300
                    },
                    velocity:{
                        x: 200,
                        y: 200
                    }
                }
            },
            text:{
                scores:{
                    combo_smash:null,
                    team1: null,
                    team2: null
                }
            }
        };
        this.actors = {
            point:{
                sound:null,
            },
            number_of_players: 2, 
            scores:{
                combo_smash:0
            },
            players: [
                {
                    image: null,
                    racket: null,
                    score: 0,
                    team: null
                },
                {
                    image: null,
                    racket: null,
                    score: 0,
                    team: null
                }
            ]
        }
        this.cursors = {
            keyboard: null,
        };

        //Socket manager
        this.socketManager = null;
    }

    preload(){

        this.load.image('court', 'assets/background/court.png');
        this.load.image('ball', 'assets/background/ball.png');
        this.load.atlas('rackets', 'assets/player/rackets.png', 'assets/player/rackets.json');    

        this.load.audio('smash', 'assets/background/smash.wav');
        this.load.audio('point', 'assets/background/point.mp3');
    }

    create() {
    
        /****AI INTEGRATION****/
        this.socketManager = new SocketManager("http://127.0.0.1:5001");
        this.input.keyboard.on(
            "keydown-LEFT",
            ()=>{
                this.socketManager.sendPlayerMove("left");
            }
        );
        this.input.keyboard.on(
            "keydown-RIGHT",
            ()=>{
                this.socketManager.sendPlayerMove("right");
            }
        );


        /****SETTINGS****/

        // Settings court
        this.background.court.image = this.add.image(400, 300, 'court');
        this.background.court.image.setOrigin(0.5, 0.5);
        this.background.court.image.setScale(1, 0.6);
    
        // Settings ball
        const initialPositionX = this.background.ball.initial.position.x;
        const initialPositionY = this.background.ball.initial.position.y;
        this.background.ball.sound = this.sound.add("smash");
        this.background.ball.image = this.physics.add.image(initialPositionX, initialPositionY, 'ball');
        this.background.ball.image.setOrigin(0.5, 0.5);
        this.background.ball.image.setScale(0.1, 0.1);
        this.background.ball.image.setCollideWorldBounds(true);
        this.background.ball.image.setBounce(1, 1);
        this.background.ball.image.setVelocity(
            this.background.ball.initial.velocity.x,
            this.background.ball.initial.velocity.y
        );

        // Settings scores
        this.background.text.scores.team1 = this.add.text(100, 20, 'Team 1: 0', { 
            fontSize: '20px', 
            fontFamily: "'Press Start 2P', 'Courier New', monospace",
            fill: 'black' 
        });
        this.background.text.scores.team2 = this.add.text(600, 20, 'Team 2: 0', { 
            fontSize: '20px', 
            fontFamily: "'Press Start 2P', 'Courier New', monospace",
            fill: 'black' 
        });

        //Combo smash
        this.background.text.scores.combo_smash = this.add.text(300,20, "Combo smash: 0",{
            fontSize:'15px',
            fontFamily: "'Press Start 2P', 'Courier New', monospace",
            fill:'green',
            backgroundColor:"black"
        });
    
        // Settings players
        this.actors.point.sound = this.sound.add("point");
        for(let countOfPlayers = 0; countOfPlayers < this.actors.number_of_players; countOfPlayers++){
            
            //Define the position of the players and the team
            let xCoordinates = 100; //By default is on the right side
            this.actors.players[countOfPlayers].team = "team1";
            if(countOfPlayers % 2 == 0){
                xCoordinates = 700; //In case of even number of players, the player is on the left side
                this.actors.players[countOfPlayers].team = "team2";
            }
            this.actors.players[countOfPlayers].team =
            this.actors.players[countOfPlayers].racket = this.physics.add.image(xCoordinates, 300, 'rackets', this.getRandomRacket());
            this.actors.players[countOfPlayers].racket.setOrigin(0.5, 0.5);
            this.actors.players[countOfPlayers].racket.setScale(0.2, 0.2);
            this.actors.players[countOfPlayers].racket.setCollideWorldBounds(true);
            this.actors.players[countOfPlayers].racket.setImmovable(true);

            //Colliders
            this.physics.add.collider(
                this.actors.players[countOfPlayers].racket, 
                this.background.ball.image,
                this.handleBallCollision,
                null,
                this
            );
        }

        //Define the cursor
        this.cursors.keyboard = this.input.keyboard.createCursorKeys();
        this.cursors.keyboard.w = this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.W);
        this.cursors.keyboard.s = this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.S);

        /****MECHANICS****/
        this.physics.world.on('worldbounds', this.handleGoal, this);
        this.background.ball.image.body.onWorldBounds = true;
    }

    update(){

        //Up and down movement
        this.actors.players[0].racket.setVelocityY(0);
        if(this.cursors.keyboard.up.isDown){
            this.actors.players[0].racket.setVelocityY(-500);
        }else if(this.cursors.keyboard.down.isDown){
            this.actors.players[0].racket.setVelocityY(500);
        }

        //Player 2 movement
        this.actors.players[1].racket.setVelocityY(0);
        if(this.cursors.keyboard.w.isDown){
            this.socketManager.sendPlayerMove("up");
            this.actors.players[1].racket.setVelocityY(-500);
        }else if(this.cursors.keyboard.s.isDown){
            this.socketManager.sendPlayerMove("down");
            this.actors.players[1].racket.setVelocityY(500);
        }

        this.background.text.scores.combo_smash.setText(
            "Combo smash: " + 
            this.actors.scores.combo_smash
        );
    }

    getRandomRacket(){
        return Phaser.Utils.Array.GetRandom(this.textures.get("rackets").getFrameNames());
    }

    handleBallCollision(racket, ball) {
        
        this.background.ball.sound.play();
        this.actors.scores.combo_smash +=1;

        // Calculate the difference between the ball's Y position and the racket's Y position
        const diff = ball.y - racket.y;
    
        // Adjust the ball's velocity based on the collision point
        ball.setVelocityY(diff * randomNormal(3.5,1.1)); // Adjust multiplier for sensitivity
    

        // Optionally, increase the ball's speed slightly to make the game more dynamic
        ball.setVelocityX(ball.body.velocity.x * randomNormal(1.5,0.4));
    
        // Update the ball's angle based on its velocity
        const angle = Math.atan2(ball.body.velocity.y, ball.body.velocity.x);
        ball.setAngle(Phaser.Math.RadToDeg(angle));
    
    }

    handleGoal(body, up, down, left, right) {

        const initialPositionX = this.background.ball.initial.position.x;
        const initialPositionY = this.background.ball.initial.position.y;
        if (left || right) {
            this.actors.point.sound.play();
            this.actors.scores.combo_smash = 0;
            // Update the score
            if (left) {
                this.actors.players[1].score++;
                this.background.text.scores.team2.setText(
                    `Team 2: ${this.actors.players[1].score}`,{
                        fontSize: '20px', 
                        fontFamily: "'Press Start 2P', 'Courier New', monospace",
                        fill: 'black' 
                    }
                );
            } else if (right) {
                this.actors.players[0].score++;
                this.background.text.scores.team1.setText(
                    `Team 1: ${this.actors.players[0].score}`,{
                        fontSize: '20px', 
                        fontFamily: "'Press Start 2P', 'Courier New', monospace",
                        fill: 'black' 
                    }
                );
            }
    
            // Reset the ball to the center
            this.background.ball.image.setPosition(initialPositionX, initialPositionY);
            this.background.ball.image.setVelocity(
                randomNormal(175, 40)* (left ? 1 : -1), 
                randomNormal(170, 25)* (left ? 1 : -1)
            );
        }
    }
}

export default BasicGame;