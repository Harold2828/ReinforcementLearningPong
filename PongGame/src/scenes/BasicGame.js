import Phaser from 'phaser';

class BasicGame extends Phaser.Scene {

    constructor() {
    
        super({ key: 'BasicGame' });

        this.background = {
            court:{
                image: null
            },
            ball:{
                image: null,
                initial:{
                    position:{
                        x: 400,
                        y: 300
                    }
                }
            },
            text:{
                scores:{
                    team1: null,
                    team2: null
                }
            }
        };
        this.actors = {
            number_of_players: 2, 
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
    }

    preload(){

        this.load.image('court', 'assets/background/court.png');
        this.load.image('ball', 'assets/background/ball.png');
        this.load.atlas('rackets', 'assets/player/rackets.png', 'assets/player/rackets.json');    }

    create() {
    
        /****SETTINGS****/

        // Settings court
        this.background.court.image = this.add.image(400, 300, 'court');
        this.background.court.image.setOrigin(0.5, 0.5);
        this.background.court.image.setScale(1, 0.6);
    
        // Settings ball
        const initialPositionX = this.background.ball.initial.position.x;
        const initialPositionY = this.background.ball.initial.position.y;
        this.background.ball.image = this.physics.add.image(initialPositionX, initialPositionY, 'ball');
        this.background.ball.image.setOrigin(0.5, 0.5);
        this.background.ball.image.setScale(0.1, 0.1);
        this.background.ball.image.setCollideWorldBounds(true);
        this.background.ball.image.setBounce(1, 1);
        this.background.ball.image.setVelocity(
            Phaser.Math.Between(-200, 200),
            Phaser.Math.Between(-200, 200) 
        );

        // Settings scores
        this.background.text.scores.team1 = this.add.text(100, 20, 'Team 1: 0', { 
            fontSize: '32px', 
            fill: 'black' 
        });
        this.background.text.scores.team2 = this.add.text(600, 20, 'Team 2: 0', { 
            fontSize: '32px', 
            fill: 'black' 
        });
    
        // Settings players
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
            this.actors.players[0].racket.setVelocityY(-200);
        }else if(this.cursors.keyboard.down.isDown){
            this.actors.players[0].racket.setVelocityY(200);
        }

        //Player 2 movement
        this.actors.players[1].racket.setVelocityY(0);
        if(this.cursors.keyboard.w.isDown){
            this.actors.players[1].racket.setVelocityY(-200);
        }else if(this.cursors.keyboard.s.isDown){
            this.actors.players[1].racket.setVelocityY(200);
        }
    }

    getRandomRacket(){
        return Phaser.Utils.Array.GetRandom(this.textures.get("rackets").getFrameNames());
    }

    handleBallCollision(ball, racket){
        const ballVelocity = ball.body.velocity;
        ball.setVelocityX(ballVelocity.x * -1.3);
        ball.setVelocityY(ballVelocity.y * -1.3);
    }

    handleGoal(body, up, down, left, right) {
        const initialPositionX = this.background.ball.initial.position.x;
        const initialPositionY = this.background.ball.initial.position.y;
        if (left || right) {
            // Update the score
            if (left) {
                this.actors.players[1].score++;
                this.background.text.scores.team2.setText(`Team 2: ${this.actors.players[1].score}`);
            } else if (right) {
                this.actors.players[0].score++;
                this.background.text.scores.team1.setText(`Team 1: ${this.actors.players[0].score}`);
            }
    
            // Reset the ball to the center
            this.background.ball.image.setPosition(initialPositionX, initialPositionY);
            this.background.ball.image.setVelocity(
                Phaser.Math.Between(-200, 200)* (left ? 1 : -1), 
                Phaser.Math.Between(-200, 200)
            );
        }
    }
}

export default BasicGame;