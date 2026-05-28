"""
Ingesta de conversaciones exportadas desde claude.ai
Formato: conversations.json dentro del ZIP de export

Uso:
    python tools/ingest_claude.py /ruta/conversations.json
    python tools/ingest_claude.py /ruta/conversations.json --dry-run
"""
import sys
import json
import argparse
sys.path.insert(0, '/opt/jarvis')

from memory.embeddings import create_embedding
from memory.store import save_memory
from tools.detect_project import detect_project

# Mínimo de caracteres para considerar un fragmento útil
MIN_CHARS = 150
# Máximo de caracteres por fragmento (para no superar límite de embedding)
MAX_CHARS = 2000


def chunk_text(text: str) -> list[str]:
    """Divide texto largo en fragmentos solapados."""
    if len(text) <= MAX_CHARS:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + MAX_CHARS
        chunks.append(text[start:end])
        start += MAX_CHARS - 200  # 200 chars de solapamiento
    return chunks


def extract_messages(conversation: dict) -> list[dict]:
    """
    Extrae mensajes de una conversación de Claude.
    Devuelve lista de {role, text}.
    """
    messages = []
    for msg in conversation.get("chat_messages", []):
        role = msg.get("sender", "unknown")
        # El contenido puede ser string o lista de bloques
        content = msg.get("text", "") or ""
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") for block in content
                if isinstance(block, dict)
            )
        if content and len(content.strip()) >= MIN_CHARS:
            messages.append({"role": role, "text": content.strip()})
    return messages


def ingest(filepath: str, dry_run: bool = False):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # El export de Claude es una lista de conversaciones
    conversations = data if isinstance(data, list) else [data]

    print(f"Conversaciones encontradas: {len(conversations)}")
    total_memories = 0

    for i, conv in enumerate(conversations):
        title = conv.get("name") or conv.get("title") or f"Conversación {i+1}"
        messages = extract_messages(conv)

        if not messages:
            continue

        # Concatenar toda la conversación para detectar proyecto
        full_text = " ".join(m["text"] for m in messages)[:1000]
        project = detect_project(full_text, threshold=0.30)
        project_id   = project["id"]   if project else None
        project_name = project["name"] if project else "sin proyecto"

        print(f"\n[{i+1}/{len(conversations)}] {title[:60]}")
        print(f"  Proyecto: {project_name} | Mensajes: {len(messages)}")

        # Ingestar solo mensajes del asistente (más densos en información)
        # y mensajes largos del usuario (decisiones, contexto)
        for msg in messages:
            if msg["role"] == "human" and len(msg["text"]) < 300:
                continue  # saltar mensajes cortos del usuario

            for chunk in chunk_text(msg["text"]):
                if len(chunk) < MIN_CHARS:
                    continue

                summary = f"{title[:50]} [{msg['role']}]"

                if not dry_run:
                    try:
                        embedding = create_embedding(chunk)
                        save_memory(
                            project_id=project_id,
                            content=chunk,
                            summary=summary,
                            embedding=embedding,
                            importance=2 if msg["role"] == "assistant" else 1,
                        )
                        total_memories += 1
                        print(f"  ✓ chunk {len(chunk)} chars → {project_name}")
                    except Exception as e:
                        print(f"  ✗ Error: {e}")
                else:
                    print(f"  [DRY] chunk {len(chunk)} chars → {project_name}")
                    total_memories += 1

    print(f"\nTotal memorias {'que se insertarían' if dry_run else 'insertadas'}: {total_memories}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="Ruta al archivo conversations.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simula sin insertar en BD")
    args = parser.parse_args()
    ingest(args.filepath, dry_run=args.dry_run)
