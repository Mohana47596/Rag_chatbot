import os
import pickle
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# GLOBAL VARIABLES
# ==============================
_embed_model = None
_llm = None
_documents = None
_embeddings = None


# ==============================
# LOAD MODELS (ONLY ONCE)
# ==============================
def load_models():
    global _embed_model, _llm

    if _embed_model is None:
        print("Loading embedding model...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    if _llm is None:
        print("Loading LLM...")
        _llm = hf_pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            max_new_tokens=200,
            do_sample=False,
            temperature=0.0
        )


# ==============================
# LOAD VECTOR STORE
# ==============================
def load_vector_store():
    global _documents, _embeddings

    if _documents is not None and _embeddings is not None:
        return

    file_path = "rag/vector_store.pkl"

    if not os.path.exists(file_path):
        raise Exception("Please upload a PDF first.")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    _documents = data["documents"]
    _embeddings = np.array(data["embeddings"])

    print("Vector store loaded successfully.")


# ==============================
# RETRIEVAL
# ==============================
def retrieve(query, top_k=3):

    if _embeddings is None or _documents is None:
        return [], []

    if len(_embeddings) == 0:
        return [], []

    query_embedding = _embed_model.encode([query])

    similarities = cosine_similarity(query_embedding, _embeddings)[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    retrieved_docs = [_documents[i] for i in top_indices]
    retrieved_scores = [float(similarities[i]) for i in top_indices]

    return retrieved_docs, retrieved_scores retrieved_docs, retrieved_scores


# ==============================
# QUESTION TYPE DETECTION
# ==============================
def is_table_question(question):

    numeric_keywords = [
        "how much",
        "how many",
        "amount",
        "total",
        "profit",
        "loss",
        "revenue",
        "sales",
        "income",
        "balance",
        "expense",
        "calculate",
        "percentage"
    ]

    return any(k in question.lower() for k in numeric_keywords)


def is_advice_question(question):

    advice_words = [
        "suggest",
        "recommend",
        "should i",
        "best investment",
        "where to invest"
    ]

    return any(w in question.lower() for w in advice_words)


def is_comparison_question(question):

    comparison_words = [
        "difference",
        "compare",
        "distinguish",
        "vs",
        "versus"
    ]

    return any(w in question.lower() for w in comparison_words)


# ==============================
# EXACT NUMERIC EXTRACTION
# ==============================
def extract_exact_numeric(context_docs, question):

    q_lower = question.lower()

    metric_aliases = {
        "revenue": ["revenue", "sales", "turnover"],
        "profit": ["profit", "net profit"],
        "tax": ["tax"],
        "interest": ["interest"],
        "dividend": ["dividend"]
    }

    requested_metric = None

    for key, aliases in metric_aliases.items():
        for alias in aliases:
            if alias in q_lower:
                requested_metric = key
                break
        if requested_metric:
            break

    if not requested_metric:
        return None

    full_text = "\n".join(context_docs)

    lines = [l.strip() for l in full_text.split("\n") if l.strip()]

    for line in lines:

        if requested_metric in line.lower():

            numbers = re.findall(r"\d[\d,\.]*", line)

            if numbers:
                return numbers[-1]

    return None


# ==============================
# REMOVE REPETITION
# ==============================
def remove_repetition(text):

    sentences = text.split(".")
    seen = set()
    cleaned = []

    for s in sentences:
        s_clean = s.strip().lower()

        if s_clean and s_clean not in seen:
            seen.add(s_clean)
            cleaned.append(s.strip())

    return ". ".join(cleaned)


# ==============================
# BUILD PROMPT
# ==============================
def build_rag_prompt(context_docs, question):

    context = "\n\n".join(context_docs)

    if is_comparison_question(question):

        return f"""
You are a finance professor.

Using ONLY the context below explain the comparison clearly.

Context:
{context}

Question:
{question}

Answer:
"""

    return f"""
You are a finance professor.

Using ONLY the context below explain clearly.

Rules:
- Minimum 5 sentences
- Include formula if relevant
- Give simple example
- No repetition

Context:
{context}

Question:
{question}

Answer:
"""


# ==============================
# MAIN RAG FUNCTION
# ==============================
def rag_answer(question: str):

    try:
        load_models()
        load_vector_store()

        context_docs, scores = retrieve(question)

        max_score = max(scores) if scores else 0

        # NUMERIC MODE
        if is_table_question(question):
            exact = extract_exact_numeric(context_docs, question)
            if exact:
                return {
                    "question": question,
                    "answer": exact,
                    "mode": "Exact-Numeric"
                }

        # ADVICE MODE
        if is_advice_question(question):

            prompt = f"""
You are a financial advisor.
Give 5 short practical tips.

Question:
{question}

Answer:
"""

            result = _llm(prompt)
            answer = remove_repetition(result[0]["generated_text"].strip())

            return {
                "question": question,
                "answer": answer,
                "mode": "Advice"
            }

        # GENERAL MODE
        if max_score < 0.30:

            prompt = f"""
Explain clearly with example.

Question:
{question}

Answer:
"""

            result = _llm(prompt)
            answer = remove_repetition(result[0]["generated_text"].strip())

            return {
                "question": question,
                "answer": answer,
                "mode": "General-LLM"
            }

        # RAG MODE
        prompt = build_rag_prompt(context_docs, question)

        result = _llm(prompt)

        answer = remove_repetition(result[0]["generated_text"].strip())

        confidence = round(max_score, 2)

        return {
            "question": question,
            "answer": answer,
            "confidence_score": confidence,
            "mode": "RAG-PDF"
        }

    except Exception as e:

        print("ERROR IN RAG PIPELINE:", str(e))

        return {
            "answer": "Internal server error in RAG pipeline",
            "error": str(e)
        }
