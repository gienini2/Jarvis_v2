import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import pytz

from memory.search import search_memories
from memory.store import (
    get_recent_memories, get_memories_by_project,
    get_project_by_name, update_project_activity,
)
from memory.embeddings import create_embedding

logger = logging.getLogger("jarvis.context_builder")

_TZ_MADRID = pytz.timezone("Europe/Madrid")

DIAS_ES = {
    "Monday": "lunes", "Tuesday": "martes", "Wednesday": "miércoles",
    "Thursday": "jueves", "Friday": "viernes", "Saturday": "sábado",
    "Sunday": "domingo",
}

# Ruta de la ficha personal
PROFILE_PATH = os.path.join(os.path.dirname(__file__), "juan_pablo_profile.json")

def _load_profile() -> dict:
    try:
        with open(PROFILE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("No se pudo cargar la ficha de perfil: %s", e)
        return {}

def _profile_summary(profile: dict) -> str:
    """Extrae un resumen compacto de la ficha para el prompt."""
    if not profile:
        return ""

    lines = []

    # Identidad básica
    id_ = profile.get("identidad", {})
    if id_:
        lines.append(f"Usuario: {id_.get('nombre')}, {id_.get('edad')} años, {id_.get('ubicacion')}")
        lines.append(f"Rol: {id_.get('profesion_actual')}")
        lines.append(f"Estado: {id_.get('estado_profesional')}")

    # Perfil cognitivo
    cog = profile.get("perfil_cognitivo", {})
    if cog:
        lines.append(f"Perfil cognitivo: IQ {cog.get('iq_estimado')}, pensamiento {cog.get('tipo_pensamiento')}")
        lines.append(f"Nota: {cog.get('nota_para_jarvis', '')}")

    # Instrucciones activas
    inst = profile.get("instrucciones_para_jarvis", {})
    if inst:
        lines.append(f"Tono: {inst.get('tono')}")
        lines.append(f"Decisiones: {inst.get('decisiones')}")
        lines.append(f"Discrepar: {inst.get('cuando_discrepar')}")

    # Proyectos activos (solo nombres y estado)
    proyectos = profile.get("proyectos_activos", {})
    if proyectos:
        resumen = ", ".join(
            f"{k} ({v.get('estado', '?')})"
            for k, v in proyectos.items()
        )
        lines.append(f"Proyectos: {resumen}")

    # Objetivo monetización
    mon = profile.get("monetizacion", {})
    if mon:
        lines.append(f"Objetivo económico: {mon.get('objetivo_ingreso')}")

    return "\n".join(lines)


@dataclass
class DetectedProject:
    project_id:   Optional[int] = None
    project_name: Optional[str] = None
    confidence:   float         = 0.0
    method:       str           = "default"


@dataclass
class RetrievedMemory:
    memory_id:  int
    content:    str
    summary:    Optional[str]      = None
    importance: int                = 1
    project:    Optional[str]      = None
    created_at: Optional[datetime] = None
    similarity: float              = 0.0
    source:     str                = "semantic"


@dataclass
class ContextPayload:
    user_message:     str
    session_id:       Optional[int]
    detected_project: DetectedProject
    memories:         list = field(default_factory=list)
    system_hints:     list = field(default_factory=list)
    profile:          dict = field(default_factory=dict)
    built_at:         datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_prompt(self):
        now_mad = datetime.now(_TZ_MADRID)
        dia_es  = DIAS_ES.get(now_mad.strftime("%A"), now_mad.strftime("%A"))
        fecha   = now_mad.strftime(f"{dia_es} %d/%m/%Y %H:%M")

        lines = ["── CONTEXTO DE JARVIS ──────────────────────────────────"]
        lines.append(f"Ahora mismo    : {fecha} (Europe/Madrid)")

        # Perfil del usuario
        if self.profile:
            perfil_txt = _profile_summary(self.profile)
            if perfil_txt:
                lines.append("\n── PERFIL ──────────────────────────────────────────────")
                lines.append(perfil_txt)

        # Proyecto activo
        lines.append("\n── SESIÓN ──────────────────────────────────────────────")
        if self.detected_project.project_name:
            lines.append(
                f"Proyecto activo: {self.detected_project.project_name} "
                f"({self.detected_project.confidence:.0%}, {self.detected_project.method})"
            )
        else:
            lines.append("Proyecto activo: no identificado")

        # Memorias
        if self.memories:
            lines.append(f"\nMemorias relevantes ({len(self.memories)}):")
            for i, m in enumerate(self.memories, 1):
                ts  = m.created_at.strftime("%Y-%m-%d") if m.created_at else "?"
                tag = f"{m.similarity:.0%}" if m.source == "semantic" else "reciente"
                lines.append(f"  {i}. [{ts}][{tag}] {m.content[:200]}")
                if m.summary:
                    lines.append(f"     -> {m.summary}")
        else:
            lines.append("\nSin memorias previas relevantes.")

        for h in self.system_hints:
            lines.append(f"  . {h}")

        lines.append("────────────────────────────────────────────────────")
        return "\n".join(lines)


class ContextBuilder:

    def __init__(self):
        self.profile = _load_profile()
        if self.profile:
            logger.info("Perfil cargado: %s v%s",
                        self.profile.get("identidad", {}).get("nombre", "?"),
                        self.profile.get("version", "?"))

    def build(self, user_message, session_id=None, active_project=None):
        hints     = []
        embedding = self._embedding(user_message)
        detected  = self._detect_project(embedding, session_project=active_project)

        if (active_project and detected.project_name
                and detected.project_name != active_project
                and detected.confidence >= 0.35):
            hints.append(
                f"Cambio de proyecto: '{active_project}' -> '{detected.project_name}'"
            )

        memories = self._retrieve(embedding, detected)
        memories = self._budget(memories)

        if detected.project_id:
            try:
                update_project_activity(detected.project_id)
            except Exception:
                pass

        logger.info("Context built | project=%s conf=%.2f memories=%d",
                    detected.project_name, detected.confidence, len(memories))

        return ContextPayload(
            user_message=user_message,
            session_id=session_id,
            detected_project=detected,
            memories=memories,
            system_hints=hints,
            profile=self.profile,
        )

    def _detect_project(self, embedding, session_project=None):
        if embedding:
            try:
                results = search_memories(
                    query_embedding=embedding,
                    top_k=15,
                    threshold=0.40,
                )
                if results:
                    votes = {}
                    names = {}
                    for r in results:
                        pid  = r.get("project_id")
                        name = r.get("project_name")
                        if pid is None:
                            continue
                        score = r["similarity"] * r.get("importance", 1)
                        votes[pid] = votes.get(pid, 0) + score
                        names[pid] = name

                    if votes:
                        best_id    = max(votes, key=votes.get)
                        total      = sum(votes.values())
                        confidence = votes[best_id] / total
                        logger.info("Project votes: %s", {
                            names.get(k, k): round(v, 2) for k, v in votes.items()
                        })
                        return DetectedProject(
                            project_id=best_id,
                            project_name=names[best_id],
                            confidence=confidence,
                            method="vote",
                        )
            except Exception as e:
                logger.warning("Project detection error: %s", e)

        if session_project:
            info = self._resolve(session_project)
            return DetectedProject(
                project_id=info.get("id"),
                project_name=info.get("name", session_project),
                confidence=0.5,
                method="session",
            )

        return DetectedProject()

    def _resolve(self, name):
        try:
            r = get_project_by_name(name)
            if r:
                return r
        except Exception:
            pass
        return {"id": None, "name": name}

    def _retrieve(self, embedding, project):
        semantic, recent = [], []

        if embedding:
            try:
                for r in search_memories(embedding, top_k=5,
                                         threshold=0.72,
                                         project_id=project.project_id):
                    semantic.append(RetrievedMemory(
                        memory_id=r["id"], content=r["content"],
                        summary=r.get("summary"), importance=r.get("importance", 1),
                        project=r.get("project_name"), created_at=r.get("created_at"),
                        similarity=r["similarity"], source="semantic",
                    ))
            except Exception as e:
                logger.warning("semantic retrieve error: %s", e)

        try:
            raw = (get_memories_by_project(project.project_id, limit=3)
                   if project.project_id else get_recent_memories(limit=3))
            for r in raw:
                recent.append(RetrievedMemory(
                    memory_id=r["id"], content=r["content"],
                    summary=r.get("summary"), importance=r.get("importance", 1),
                    project=r.get("project_name"), created_at=r.get("created_at"),
                    similarity=0.0, source="recent",
                ))
        except Exception as e:
            logger.warning("recent retrieve error: %s", e)

        seen, combined = set(), []
        for m in semantic + recent:
            if m.memory_id not in seen:
                seen.add(m.memory_id)
                combined.append(m)

        combined.sort(key=lambda m: (
            -(m.similarity if m.source == "semantic" else 0),
            -m.importance,
            -(m.created_at.timestamp() if m.created_at else 0),
        ))
        return combined

    def _embedding(self, text):
        try:
            return create_embedding(text)
        except Exception as e:
            logger.warning("embedding error: %s", e)
            return None

    def _budget(self, memories, max_chars=6000):
        used, kept = 0, []
        for m in memories:
            size = len(m.content) + len(m.summary or "")
            if used + size > max_chars:
                break
            kept.append(m)
            used += size
        return kept


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    builder = ContextBuilder()
    for msg in [
        "analiza el tingbot para XAUUSD",
        "cuanto me queda del temario de oposiciones",
        "añade memoria semantica al orquestador",
    ]:
        print(f"\n{'='*55}\nMSG: {msg}")
        try:
            p = builder.build(msg)
            print(p.to_prompt())
        except Exception as e:
            print(f"ERROR: {e}")
