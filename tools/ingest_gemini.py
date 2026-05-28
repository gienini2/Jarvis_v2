import sys, json, argparse
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

def ingest(filepath, dry_run=False):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalizar: puede ser lista plana o lista de conversaciones
    if isinstance(data, list) and len(data) > 0:
        first = data[0]
        # Lista plana de mensajes {role, text}
        if "role" in first and "text" in first:
            conversations = [{"title": filepath.split("/")[-1], "messages": data}]
        # Lista de conversaciones con mensajes dentro
        elif "messages" in first:
            conversations = data
        else:
            conversations = [{"title": filepath.split("/")[-1], "messages": data}]
    else:
        print(f"Formato no reconocido en {filepath}")
        return 0

    print(f"Conversaciones: {len(conversations)}")
    total = 0

    for i, conv in enumerate(conversations):
        title    = conv.get("title", f"conv_{i+1}")
        messages = conv.get("messages", conv.get("chat_messages", []))

        # Filtrar mensajes válidos
        valid = [m for m in messages
                 if isinstance(m, dict)
                 and m.get("role") in ("user", "assistant", "human")
                 and len(m.get("text", "")) >= MIN_CHARS]

        if not valid:
            continue

        # Detectar proyecto con los primeros 1000 chars
        sample  = " ".join(m["text"] for m in valid[:5])[:1000]
        project = detect_project(sample, threshold=0.30)
        pid     = project["id"]   if project else None
        pname   = project["name"] if project else "sin proyecto"

        print(f"\n[{i+1}] {title[:60]} | {pname} | {len(valid)} msgs")

        for msg in valid:
            if msg["role"] in ("user", "human") and len(msg["text"]) < 300:
                continue
            for chunk in chunk_text(msg["text"]):
                if len(chunk) < MIN_CHARS:
                    continue
                if not dry_run:
                    try:
                        emb = create_embedding(chunk)
                        save_memory(
                            project_id=pid,
                            content=chunk,
                            summary=f"{title[:50]} [{msg['role']}]",
                            embedding=emb,
                            importance=2 if msg["role"] == "assistant" else 1,
                        )
                        total += 1
                    except Exception as e:
                        print(f"  ERROR: {e}")
                else:
                    total += 1

    print(f"\nTotal {'simuladas' if dry_run else 'insertadas'}: {total}")
    return total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ingest(args.filepath, dry_run=args.dry_run)
