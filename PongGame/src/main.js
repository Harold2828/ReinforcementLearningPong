import Phaser from "phaser";
import BasicGame from "./scenes/BasicGame";

const config = {
    type: Phaser.AUTO,
    width: 800,
    height: 600,
    backgroundColor: "#000000",
    parent: "game-container",
    physics: {
        default: "arcade",
        arcade: {
            gravity: { y: 0 },
            debug: false,
        },
    },
    scene: BasicGame,
};
const game = new Phaser.Game(config);