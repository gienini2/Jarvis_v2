from memory.embeddings import create_embedding
from memory.store import save_memory
from memory.search import search_memories

text = "Tingbot es mi sistema de trading automático."

embedding = create_embedding(text)

save_memory(
    project_id=1,
    content=text,
    summary="descripcion del tingbot",
    embedding=embedding
)

results = search_memories(embedding)

print(results)
