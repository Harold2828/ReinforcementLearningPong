# Pong Evolution Technical Documentation

This documentation describes the current `genetic-algorithm` working tree.

## Documents

- [Architecture](ARCHITECTURE.md): runtime components, simulation, Socket.IO, and persistence.
- [ML and evolution](ML_EVOLUTION.md): DQN, fitness, evaluation, inheritance, and generation lifecycle.
- [Developer guide](DEVELOPER_GUIDE.md): modification points, configuration, Docker, Makefile, tests, and limitations.

## System overview

```mermaid
flowchart TB
    UI[Phaser EvolutionTrainingScene] <-->|Socket.IO controls and LIVE batches| S[Flask-SocketIO]
    S --> O[EvolutionTrainingService]
    O --> A[12 independent DQNAgent instances]
    O --> P[6 independent MatchSession arenas]
    O --> E[RoundRobinEvaluator]
    O --> G[GeneticAlgorithm]
    O --> DB[(SQLite EvolutionStore)]
    O --> C[(Versioned checkpoints)]
    P --> S
    S --> UI
```

The backend owns evolution simulation and learning. The frontend never fabricates training gameplay: it validates, buffers, and interpolates real snapshots. Classic 1v1 remains a separate Phaser-owned mode.
