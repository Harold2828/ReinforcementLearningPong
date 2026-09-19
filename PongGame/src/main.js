import Phaser from "phaser";
import BasicGame from "./scenes/BasicGame";
import EvolutionTrainingScene from "./scenes/EvolutionTrainingScene";
import { EVOLUTION_VIEWS } from "./evolution/evolutionContract";
import { MockEvolutionFeed } from "./evolution/mockEvolutionFeed";
import { LiveEvolutionFeed } from "./evolution/liveEvolutionFeed";
import { EvolutionDashboard } from "./evolution/dashboard";
import SocketManager from "./utils/socketManager";
import { GAME_MODES } from "./utils/gameModeManager";

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
    scale: {
        mode: Phaser.Scale.FIT,
        autoCenter: Phaser.Scale.CENTER_BOTH,
    },
    scene: [BasicGame, EvolutionTrainingScene],
};
const game = new Phaser.Game(config);

const backendUrl = import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:5001";
const companionSocket = new SocketManager(backendUrl);
const feed = new LiveEvolutionFeed({
    sensor: companionSocket,
    fallback: new MockEvolutionFeed(),
});
feed.start();

const dashboard = new EvolutionDashboard(document.querySelector('[data-dash-role="dashboard"]'));
feed.subscribe((event) => dashboard.handleEvent(event));

function runningSceneKey() {
    const running = game.scene.getScenes(true);
    return running.length > 0 ? running[0].scene.key : null;
}

function runScene(key) {
    const current = runningSceneKey();
    if (current === key) {
        return;
    }
    if (current) {
        game.scene.stop(current);
    }
    game.scene.start(key, key === "EvolutionTraining" ? { feed } : undefined);
}

function showView(view, modeSelect) {
    const isClassicLayout = view === EVOLUTION_VIEWS.CLASSIC || view === EVOLUTION_VIEWS.HUMAN_VS_CHAMPION;
    const trainingPanel = document.querySelector(".training-panel");
    const gameContainer = document.querySelector("#game-container");
    const dashboardSection = document.querySelector('[data-dash-role="dashboard"]');

    if (trainingPanel) {
        trainingPanel.hidden = !isClassicLayout;
    }
    gameContainer.hidden = view === EVOLUTION_VIEWS.EVOLUTION_DASHBOARD;
    if (dashboardSection) {
        dashboardSection.hidden = view !== EVOLUTION_VIEWS.EVOLUTION_DASHBOARD;
    }

    runScene(view === EVOLUTION_VIEWS.EVOLUTION_TRAINING ? "EvolutionTraining" : "BasicGame");

    if (view === EVOLUTION_VIEWS.HUMAN_VS_CHAMPION && modeSelect) {
        modeSelect.value = GAME_MODES.HUMAN_VS_CHAMPION;
        modeSelect.dispatchEvent(new Event("change", { bubbles: true }));
    }
}

const modeSelect = document.querySelector('[data-action="select-game-mode"]');
document.querySelectorAll('[data-action="switch-view"]').forEach((button) => {
    button.addEventListener("click", () => {
        const view = button.dataset.view;
        if (Object.values(EVOLUTION_VIEWS).includes(view)) {
            showView(view, modeSelect);
        }
    });
});