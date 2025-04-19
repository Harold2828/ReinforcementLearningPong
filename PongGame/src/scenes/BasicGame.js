import Phaser from 'phaser';

class BasicGame extends Phaser.Scene {

    constructor() {
    
        super({ key: 'BasicGame' });

        this.background = {
            court:{
                image: null
            },
            ball:{
                image: null
            }
        };
        this.players = [
            {
                image:null
            },
            {
                image:null
            }
        ];
    }

    preload(){

        this.add.image('court', 'assets/background/court.png');
        this.add.image('ball', 'assets/background/ball.png');
        this.add.atlas('player', 'assets/player/rackets.png', 'assets/player/player.json');

    }

    create(){}

    update(){}

}

export default BasicGame;