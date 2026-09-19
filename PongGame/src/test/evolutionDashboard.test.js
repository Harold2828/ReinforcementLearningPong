/**
 * @vitest-environment jsdom
 */
import { describe, expect, it } from "vitest";
import { EvolutionDashboard } from "../evolution/dashboard";
import { EVOLUTION_EVENT_TYPES, SOURCE_LABEL } from "../evolution/evolutionContract";

function renderDashboard() {
    document.body.innerHTML = `
        <section data-dash-role="dashboard">
            <strong data-dash-role="status">Waiting</strong>
            <span data-dash-role="source"></span>
            <table><tbody data-dash-role="population"></tbody></table>
            <div data-dash-role="champion">none</div>
            <ul data-dash-role="events"></ul>
        </section>
    `;
    return new EvolutionDashboard(document.querySelector('[data-dash-role="dashboard"]'));
}

function populationEvent() {
    const agents = Array.from({ length: 10 }, (_, index) => ({
        agentId: `agent-${index}`,
        generation: 0,
        role: index < 2 ? "elite" : "offspring",
        architecture: { layerCount: (index % 4) + 1, widths: [32, 64] },
        fitness: { winRate: 0.5, pointDifferential: 0.25, combo: 0.1 },
        lineage: index < 2 ? [] : [index - 2, index - 1],
        status: "available",
    }));
    return { type: EVOLUTION_EVENT_TYPES.POPULATION, runId: "run-1", generationId: "generation-1", agents, source: SOURCE_LABEL.MOCK };
}

describe("EvolutionDashboard", () => {
    it("renders the population table", () => {
        const dashboard = renderDashboard();
        dashboard.handleEvent(populationEvent());
        expect(document.querySelector('[data-dash-role="population"]').children).toHaveLength(10);
        expect(document.querySelector('[data-dash-role="source"]').textContent).toContain("MOCK");
        expect(document.querySelector('[data-dash-role="status"]').textContent).toContain("10 agents");
    });

    it("records the champion promotion", () => {
        const dashboard = renderDashboard();
        dashboard.handleEvent(populationEvent());
        dashboard.handleEvent({
            type: EVOLUTION_EVENT_TYPES.CHAMPION_PROMOTION,
            runId: "run-1",
            agentId: "agent-0",
            checkpointHash: "mock-hash",
            evidence: { benchmark: "mock-bench-v1", fitness: { winRate: 0.9 } },
            source: SOURCE_LABEL.MOCK,
        });
        expect(document.querySelector('[data-dash-role="champion"]').textContent).toContain("agent-0");
        expect(document.querySelector('[data-dash-role="champion"]').textContent).toContain("mock-hash");
    });

    it("surfaces controlled errors", () => {
        const dashboard = renderDashboard();
        dashboard.handleEvent({
            type: EVOLUTION_EVENT_TYPES.CONTROLLED_ERROR,
            runId: "run-1",
            code: "MOCK_NOT_IMPLEMENTED",
            message: "pending",
            source: SOURCE_LABEL.MOCK,
        });
        expect(document.querySelector('[data-dash-role="status"]').textContent).toContain("MOCK_NOT_IMPLEMENTED");
        expect(document.querySelector('[data-dash-role="events"]').children).toHaveLength(1);
    });

    it("binds evaluation results onto existing fitness cells", () => {
        const dashboard = renderDashboard();
        dashboard.handleEvent(populationEvent());
        dashboard.handleEvent({
            type: EVOLUTION_EVENT_TYPES.EVALUATION,
            runId: "run-1",
            generationId: "generation-1",
            benchmark: "mock-bench-v1",
            results: [{ agentId: "agent-0", fitness: { winRate: 0.99, pointDifferential: 0.9, combo: 0.8 } }],
            source: SOURCE_LABEL.MOCK,
        });
        const firstRowCells = [...document.querySelectorAll('[data-dash-role="population"] tr')][0].children;
        expect(firstRowCells[4].textContent).toContain("winRate: 0.99");
    });
});