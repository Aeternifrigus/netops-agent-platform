"""FastAPI wrapper around the tools and agent construction."""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agents import tools
from .agents.orchestrator import build_orchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NetOps Agent Platform",
    description="Multi-agent network operations assistant: topology graph, "
                "anomaly scoring, and incident-relevance ranking.",
    version="0.1.0",
)

_orchestrator = None


def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = build_orchestrator()
    return _orchestrator


# ── schemas ──────────────────────────────────────────────────────

class ImpactRequest(BaseModel):
    tower_id: str
    max_hops: int = Field(default=3, ge=1, le=10)


class HealthRequest(BaseModel):
    latency_ms: float = Field(ge=0)
    packet_loss_pct: float = Field(ge=0, le=100)
    retransmit_rate: float = Field(ge=0)
    connected_devices: float = Field(ge=0)


class RelevanceRequest(BaseModel):
    event_descriptions: list[str]
    focus_index: int = Field(ge=0)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


# ── direct tool endpoints (no LLM required) ──────────────────────

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_configured": bool(os.environ.get("GOOGLE_API_KEY")),
        "graph_backend": tools._get_graph().name if tools._graph else "not yet loaded",
    }


@app.post("/tools/impact")
def impact(req: ImpactRequest) -> dict:
    try:
        return tools.get_downstream_impact(req.tower_id, max_hops=req.max_hops)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/tools/health-score")
def health_score(req: HealthRequest) -> dict:
    return tools.score_tower_health(
        req.latency_ms, req.packet_loss_pct, req.retransmit_rate, req.connected_devices
    )


@app.post("/tools/relevance")
def relevance(req: RelevanceRequest) -> dict:
    try:
        return tools.rank_relevant_events(req.event_descriptions, req.focus_index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ── live agent endpoint (requires a model key) ───────────────────

@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    if not os.environ.get("GOOGLE_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail=(
                "No model credentials configured. Set GOOGLE_API_KEY to run a live "
                "agent conversation, or use the /tools/* endpoints, which need no "
                "credentials."
            ),
        )

    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=_get_orchestrator(), app_name="netops")
    session = await runner.session_service.create_session(
        app_name="netops", user_id="api-user"
    )

    reply_text = ""
    async for event in runner.run_async(
        user_id="api-user",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=req.message)]),
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    reply_text += part.text

    return {"reply": reply_text}
