# backend/agents/retriever_agent.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_top_chunks(question, chunks, embed_model, top_k=4):
    question_emb = embed_model.encode([question])
    chunk_embs = embed_model.encode(chunks)

    scores = cosine_similarity(question_emb, chunk_embs)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [chunks[i] for i in top_indices]
