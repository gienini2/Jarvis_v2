"""
Detección de proyecto por similitud semántica contra embeddings de proyectos.
Sustituye completamente la detección por keywords.
"""
import sys
sys.path.insert(0, '/opt/jarvis')

from memory.db import get_connection, release_connection
from memory.embeddings import create_embedding

def detect_project(message: str, threshold: float = 0.30) -> dict | None:
    """
    Compara el embedding del mensaje contra los embeddings de proyectos.
    Devuelve el proyecto más similar o None si no supera el threshold.
    """
    embedding = create_embedding(message)
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, description,
                   1 - (embedding <-> %s::vector) / 2 AS similarity
            FROM projects
            WHERE status = 'active' AND embedding IS NOT NULL
            ORDER BY embedding <-> %s::vector
            LIMIT 3
            """,
            (embedding, embedding),
        )
        rows = cur.fetchall()
        cur.close()

        if not rows:
            return None

        best = rows[0]
        sim  = float(best[3])

        print(f"\nTop 3 proyectos para: '{message}'")
        for r in rows:
            print(f"  {r[1]:20s} {float(r[3]):.2%}")

        if sim >= threshold:
            return {"id": best[0], "name": best[1], "similarity": sim}
        return None

    finally:
        release_connection(conn)


if __name__ == "__main__":
    tests = [
        "analiza el tingbot para XAUUSD en MT5",
        "audita sherlock y hazlo más robusto",
        "cuánto me queda del temario de oposiciones",
        "continúa el plan para llegar a 100k al año",
        "tengo que preparar el juicio por acoso",
        "cuánto peso perdí este mes",
        "añade memoria semántica al orquestador",
    ]
    for msg in tests:
        result = detect_project(msg)
        print(f"  → {result['name'] if result else 'sin proyecto'}\n")
