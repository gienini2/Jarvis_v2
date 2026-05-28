"""
Ingesta de conversaciones exportadas desde ChatGPT.
El ZIP contiene conversations.json con formato distinto al de Claude.

Uso:
    python tools/ingest_chatgpt.py /ruta/conversations.json
    python tools/ingest_chatgpt.py /ruta/conversations.json --dry-run
"""
import sys
import json
import argparse
sys.path.insert(0, '/opt/jarvis')

from memory.embeddings import create_embedding
from memory.store import save_memory
from tools.detect_project import detect_project

MIN_CHARS = 150
MAX_CHARS = 2000

def chunk_text(text):
    if len(text) <= MAX_CHARS:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + MAX_CHARS])
        start += MAX_CHARS - 200
    return chunks

def extract_messages_chatgpt(conversation):
    """
    En ChatGPT el formato es:
    conversation.mapping = { id: { message: { role, content: { parts: [] } } } }
    """
    messages = []
    mapping = conversation.get("mapping", {})
    for node in mapping.values():
        msg = node.get("message")
        if not msg:
            continue
        role = msg.get("author", {}).get("role", "unknown")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content", {})
        parts = content.get("parts", [])
        text = " ".join(p for p in parts if isinstance(p, str)).strip()
        if len(text) >= MIN_CHARS:
            messages.append({"role": role, "text": text})
    return messages

def ingest(filepath, dry_run=False):
    with open(filepath, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    print(f"Conversaciones encontradas: {len(conversations)}")
    total = 0

    for i, conv in enumerate(conversations):
        title    = conv.get("title") or f"Conversación {i+1}"
        messages = extract_messages_chatgpt(conv)
        if not messages:
            continue

        full_text  = " ".join(m["text"] for m in messages)[:1000]
        project    = detect_project(full_text, threshold=0.30)
        project_id = project["id"]   if project else None
        proj_name  = project["name"] if project else "sin proyecto"

        print(f"\n[{i+1}/{len(conversations)}] {title[:60]}")
        print(f"  Proyecto: {proj_name} | Mensajes: {len(messages)}")

        for msg in messages:
            if msg["role"] == "user" and len(msg["text"]) < 300:
                continue
            for chunk in chunk_text(msg["text"]):
                if len(chunk) < MIN_CHARS:
                    continue
                if not dry_run:
                    try:
                        emb = create_embedding(chunk)
                        save_memory(
                            project_id=project_id,
                            content=chunk,
                            summary=f"{title[:50]} [{msg['role']}]",
                            embedding=emb,
                            importance=2 if msg["role"] == "assistant" else 1,
                        )
                        total += 1
                        print(f"  ✓ {len(chunk)} chars -> {proj_name}")
                    except Exception as e:
                        print(f"  ✗ {e}")
                else:
                    print(f"  [DRY] {len(chunk)} chars -> {proj_name}")
                    total += 1

    print(f"\nTotal memorias {'simuladas' if dry_run else 'insertadas'}: {total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ingest(args.filepath, dry_run=args.dry_run)
