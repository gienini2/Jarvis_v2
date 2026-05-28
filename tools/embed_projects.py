"""
Genera embeddings para todos los proyectos y los guarda en BD.
Ejecutar una vez, y cada vez que añadas un proyecto nuevo.
"""
import sys
sys.path.insert(0, '/opt/jarvis')

from memory.db import get_connection, release_connection
from memory.embeddings import create_embedding

def embed_projects():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, description FROM projects WHERE status = 'active'")
    projects = cur.fetchall()

    for pid, name, desc in projects:
        text = f"{name}: {desc}"
        print(f"Generando embedding para '{name}'...")
        embedding = create_embedding(text)
        cur.execute(
            "UPDATE projects SET embedding = %s WHERE id = %s",
            (embedding, pid)
        )
        print(f"  ✓ {name}")

    conn.commit()
    cur.close()
    release_connection(conn)
    print("\nTodos los proyectos tienen embedding.")

if __name__ == "__main__":
    embed_projects()
