"""Multi-agent network-operations assistant, built on ADK's Agent and sub_agents."""
from __future__ import annotations

import os

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from . import tools

DEFAULT_MODEL = os.environ.get("ADK_MODEL", "gemini-2.0-flash")


def build_topology_agent(model: str = DEFAULT_MODEL) -> Agent:
    return Agent(
        name="topology_agent",
        model=model,
        description="Answers questions about network topology and outage impact.",
        instruction=(
            "You answer questions about which towers depend on which others. "
            "Use get_downstream_impact to trace what would be affected if a "
            "given tower failed. Be specific about tower ids and counts."
        ),
        tools=[FunctionTool(tools.get_downstream_impact)],
    )


def build_health_agent(model: str = DEFAULT_MODEL) -> Agent:
    return Agent(
        name="health_agent",
        model=model,
        description="Scores whether a tower's telemetry reading looks anomalous.",
        instruction=(
            "You assess whether a tower's current telemetry looks anomalous. "
            "Use score_tower_health with the reported metrics. Explain the "
            "verdict in plain terms, not just the probability."
        ),
        tools=[FunctionTool(tools.score_tower_health)],
    )


def build_incident_agent(model: str = DEFAULT_MODEL) -> Agent:
    return Agent(
        name="incident_agent",
        model=model,
        description="Ranks recent events by relevance to a current incident.",
        instruction=(
            "You help an operator understand which recent events likely relate "
            "to a current incident. Use rank_relevant_events and summarise the "
            "top few, not the full ranked list."
        ),
        tools=[FunctionTool(tools.rank_relevant_events)],
    )


def build_orchestrator(model: str = DEFAULT_MODEL) -> Agent:
    """
    The root agent. ADK automatically gives a parent agent the ability to transfer a turn to
    any of its sub_agents when the model judges that the request matches that sub-agent's
    description -- this project does not implement that routing logic itself.
    """
    topology = build_topology_agent(model)
    health = build_health_agent(model)
    incident = build_incident_agent(model)

    return Agent(
        name="netops_orchestrator",
        model=model,
        description="Routes network-operations questions to the right specialist.",
        instruction=(
            "You are the entry point for a network operations assistant. "
            "Decide whether a question is about topology/impact, tower health, "
            "or incident context, and transfer to the matching specialist "
            "agent. If a question needs more than one specialist, ask them in "
            "sequence and combine the results yourself."
        ),
        sub_agents=[topology, health, incident],
    )
