import logging
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import pytz
import httpx

from context_builder import ContextBuilder
from memory.store import save_memory, list_projects
from memory.embeddings import create_embedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis.api")

app = FastAPI(title="Jarvis AI Orchestrator", version="0.3.0")

builder = ContextBuilder()

_TZ_MADRID = pytz.timezone("Europe/Madrid")

# ── Claves de IA ──────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")


# ── Modelos ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:        str
    session_id:     Optional[int] = None
    active_project: Optional[str] = None
    model:          Optional[str] = None   # forzar modelo concreto (opcional)
    save:           bool          = True   # guardar conversación en BD

class ChatResponse(BaseModel):
    reply:           str
    project_name:    Optional[str]
    project_confidence: float
    memories_count:  int
    model_used:      str
    hints:           list[str]

class ContextRequest(BaseModel):
    message:        str
    session_id:     Optional[int] = None
    active_project: Optional[str] = None

class ContextResponse(BaseModel):
    context_prompt:     str
    project_name:       Optional[str]
    project_confidence: float
    memories_count:     int
    hints:              list[str]


# ── Endpoints de estado ───────────────────────────────────────────────────────

@app.get("/")
def root():
    now = datetime.now(_TZ_MADRID)
    return {
        "jarvis":  "online",
        "version": "0.3.0",
        "fecha": now.strftime("%d/%m/%Y %H:%M"),
    }

@app.get("/now")
def now():
    n = datetime.now(_TZ_MADRID)
    return {
        "fecha":     n.strftime("%Y-%m-%d"),
        "hora":      n.strftime("%H:%M"),
        "dia":       n.strftime("%A"),
        "timestamp": n.isoformat(),
    }

@app.get("/projects")
def projects():
    try:
        return {"projects": list_projects()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories/recent")
def recent_memories(limit: int = 10):
    from memory.store import get_recent_memories
    try:
        return {"memories": get_recent_memories(limit=limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoint de contexto (sin llamada a IA) ───────────────────────────────────

@app.post("/context", response_model=ContextResponse)
def get_context(req: ContextRequest):
    try:
        payload = builder.build(
            user_message=req.message,
            session_id=req.session_id,
            active_project=req.active_project,
        )
        return ContextResponse(
            context_prompt=payload.to_prompt(),
            project_name=payload.detected_project.project_name,
            project_confidence=payload.detected_project.confidence,
            memories_count=len(payload.memories),
            hints=payload.system_hints,
        )
    except Exception as e:
        logger.error("Error building context: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoint principal: /chat ─────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Flujo completo:
      1. Construir contexto (perfil + memorias + proyecto)
      2. Llamar a la IA (Claude por defecto)
      3. Guardar pregunta y respuesta en BD
      4. Devolver respuesta
    """
    try:
        # 1. Contexto
        payload = builder.build(
            user_message=req.message,
            session_id=req.session_id,
            active_project=req.active_project,
        )

        context_prompt = payload.to_prompt()
        model_used     = req.model or "claude"

        # 2. Llamar a la IA
        reply = await _call_ai(
            model=model_used,
            system=context_prompt,
            user_message=req.message,
        )

        # 3. Guardar en BD si save=True
        if req.save and payload.detected_project.project_id:
            pid = payload.detected_project.project_id
            try:
                # Guardar pregunta del usuario
                emb_q = create_embedding(req.message)
                save_memory(
                    project_id=pid,
                    content=req.message,
                    summary=f"[user] {req.message[:60]}",
                    embedding=emb_q,
                    importance=1,
                )
                # Guardar respuesta de Jarvis
                emb_r = create_embedding(reply)
                save_memory(
                    project_id=pid,
                    content=reply,
                    summary=f"[jarvis/{model_used}] {req.message[:60]}",
                    embedding=emb_r,
                    importance=2,
                )
                logger.info("Conversación guardada en proyecto %s", pid)
            except Exception as e:
                logger.warning("Error guardando conversación: %s", e)

        return ChatResponse(
            reply=reply,
            project_name=payload.detected_project.project_name,
            project_confidence=payload.detected_project.confidence,
            memories_count=len(payload.memories),
            model_used=model_used,
            hints=payload.system_hints,
        )

    except Exception as e:
        logger.error("Error en /chat: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Router de modelos ─────────────────────────────────────────────────────────

async def _call_ai(model: str, system: str, user_message: str) -> str:
    """
    Llama a la IA según el modelo solicitado.
    Por defecto: Claude (Anthropic).
    """
    if model in ("claude", "anthropic"):
        return await _call_claude(system, user_message)
    elif model in ("gpt", "openai", "chatgpt"):
        return await _call_openai(system, user_message)
    else:
        # Fallback a Claude
        return await _call_claude(system, user_message)


async def _call_claude(system: str, user_message: str) -> str:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY no configurada")

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-sonnet-4-20250514",
                "max_tokens": 2048,
                "system":     system,
                "messages":   [{"role": "user", "content": user_message}],
            },
        )
        r.raise_for_status()
        data = r.json()
        return data["content"][0]["text"]


async def _call_openai(system: str, user_message: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY no configurada")

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":    "gpt-4o",
                "messages": [
                    {"role": "system",  "content": system},
                    {"role": "user",    "content": user_message},
                ],
            },
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
