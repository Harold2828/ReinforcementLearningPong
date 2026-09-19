import { EVOLUTION_EVENT_TYPES, SOURCE_LABEL } from "./evolutionContract";

/**
 * EVOLUTION_DASHBOARD view controller. Pure DOM manipulation so it is
 * jsdom-testable and independent of the Phaser canvas. Renders population,
 * architectures, fitness components, lineage, champion identity, run status,
 * and event/error states from feed events.
 */

function formatFitness(fitness) {
    if (!fitness) {
        return "—";
    }
    return Object.entries(fitness)
        .map(([key, value]) => `${key}: ${Number(value).toFixed(2)}`)
        .join(" | ");
}

class EvolutionDashboard {
    constructor(root) {
        this.root = root;
        this.populationBody = root.querySelector('[data-dash-role="population"]');
        this.status = root.querySelector('[data-dash-role="status"]');
        this.source = root.querySelector('[data-dash-role="source"]');
        this.champion = root.querySelector('[data-dash-role="champion"]');
        this.eventsList = root.querySelector('[data-dash-role="events"]');
        this.agents = [];
    }

    handleEvent(event) {
        switch (event?.type) {
            case EVOLUTION_EVENT_TYPES.POPULATION:
                this.renderPopulation(event);
                break;
            case EVOLUTION_EVENT_TYPES.EVALUATION:
                this.renderEvaluation(event);
                break;
            case EVOLUTION_EVENT_TYPES.CHAMPION_PROMOTION:
                this.renderChampion(event);
                break;
            case EVOLUTION_EVENT_TYPES.CONTROLLED_ERROR:
                this.renderError(event);
                break;
            case EVOLUTION_EVENT_TYPES.MATCH_SNAPSHOT:
                this.updateRunStatus(event);
                break;
            default:
                break;
        }
    }

    renderPopulation(event) {
        this.agents = event.agents;
        this.source.textContent = `Source: ${event.source ?? SOURCE_LABEL.MOCK} | run ${event.runId} | ${event.generationId}`;
        this.setStatus(`Population loaded (${event.agents.length} agents)`);
        this.populationBody.replaceChildren(
            ...event.agents.map((agent) => {
                const row = document.createElement("tr");
                row.innerHTML = [
                    `<td>${agent.agentId}</td>`,
                    `<td>${agent.role}</td>`,
                    `<td>${agent.architecture.layerCount}</td>`,
                    `<td>${agent.architecture.widths.join(", ")}</td>`,
                    `<td>${formatFitness(agent.fitness)}</td>`,
                    `<td>${Array.isArray(agent.lineage) && agent.lineage.length ? agent.lineage.join(" ← ") : "seed"}</td>`,
                    `<td>${agent.status}</td>`,
                ].join("");
                return row;
            }),
        );
        this.pushEvent(`population event: ${event.agents.length} members`);
    }

    renderEvaluation(event) {
        this.setStatus("Evaluation results received");
        for (const result of event.results ?? []) {
            const row = [...this.populationBody.querySelectorAll("tr")].find((entry) =>
                entry.firstElementChild?.textContent === result.agentId,
            );
            if (row?.children[4]) {
                row.children[4].textContent = formatFitness(result.fitness);
            }
        }
        this.pushEvent(`evaluation event: ${event.benchmark ?? "benchmark"} (${event.results?.length ?? 0} results)`);
    }

    renderChampion(event) {
        this.champion.textContent = [
            `Champion: ${event.agentId}`,
            `checkpoint: ${event.checkpointHash ?? "—"}`,
            `fitness: ${formatFitness(event.evidence?.fitness)}`,
        ].join(" | ");
        this.setStatus(`Champion promotion: ${event.agentId}`);
        this.pushEvent(`champion promotion: ${event.agentId}`);
    }

    renderError(event) {
        this.setStatus(`${event.code ?? "error"} — ${event.message ?? ""}`);
        this.pushEvent(`controlled error: ${event.code ?? "unknown"} (${event.source ?? "unknown"})`);
    }

    updateRunStatus(event) {
        this.setStatus(`run ${event.runId} | elapsed steps ${event.elapsedSteps} | arena ${event.arenaId} ${event.status}`);
    }

    setStatus(text) {
        this.status.textContent = text;
    }

    pushEvent(text) {
        const item = document.createElement("li");
        item.textContent = text;
        this.eventsList.prepend(item);
        while (this.eventsList.children.length > 20) {
            this.eventsList.lastElementChild.remove();
        }
    }
}

export { EvolutionDashboard };