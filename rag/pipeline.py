import os
import pickle
import re
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline as hf_pipeline
from sklearn.metrics.pairwise import cosine_similarity


# ==============================
# GLOBAL VARIABLES
# ==============================

_embed_model = None
_llm = None
_documents = None
_embeddings = None
_reranker = None


# ==============================
# LOAD MODELS (ONLY ONCE)
# ==============================

def load_models():
    global _embed_model, _llm, _reranker

    if _embed_model is None:
        print("Loading embedding model...")
        _embed_model = SentenceTransformer("all-mpnet-base-v2")

    _llm = hf_pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_new_tokens=300,
        do_sample=False,
        temperature=0.0
    )

    _reranker = None


# ==============================
# LOAD VECTOR STORE (ONLY ONCE)
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

def retrieve(query, top_k=3, initial_k=8, max_chars=3000):

    query_embedding = _embed_model.encode([query])
    similarities = cosine_similarity(query_embedding, _embeddings)[0]

    initial_indices = similarities.argsort()[-initial_k:][::-1]
    candidate_docs = [_documents[i] for i in initial_indices]

    # Rerank
    reranked_indices = initial_indices[:top_k]

    retrieved_docs = []
    retrieved_scores = []
    total_chars = 0

    


# ==============================
# QUESTION TYPE DETECTION
# ==============================

def is_table_question(question):
    numeric_keywords = [
        "how much", "how many", "amount", "total",
        "profit", "loss", "revenue", "sales",
        "income", "balance", "expense",
        "year ended", "calculate", "percentage",
        "share capital", "capital", "balance sheet"
    ]
    return any(keyword in question.lower() for keyword in numeric_keywords)


def is_advice_question(question):
    advice_words = [
        "suggest", "recommend", "should i",
        "where to invest", "which stock",
        "best investment"
    ]
    return any(word in question.lower() for word in advice_words)


def is_comparison_question(question):
    comparison_words = ["difference", "compare", "distinguish", "vs", "versus"]
    return any(word in question.lower() for word in comparison_words)


# ==============================
# EXACT NUMERIC EXTRACTION
# ==============================
"""
def extract_exact_numeric(context_docs, question):

    q_lower = question.lower()

    # 1️⃣ detect requested metric
    metric_keywords = [
        "share capital",
        "operating profit",
        "net sales",
        "revenue",
        "profit",
        "income",
        "loss",
        "interest",
        "tax",
        "dividend",
        "retained earnings",
        "cost of goods sold"
    ]

    requested_metric = None
    for m in metric_keywords:
        if m in q_lower:
            requested_metric = m
            break

    if not requested_metric:
        return None

    # 2️⃣ detect requested year like 20X6
    year_match = re.search(r"20x\d+", q_lower)
    requested_year = year_match.group() if year_match else None

    for doc in context_docs:

        lines = [l.strip() for l in doc.split("\n") if l.strip()]

        # 3️⃣ find header row containing years
        header_line = None
        for line in lines:
            if re.search(r"20x\d+", line.lower()):
                header_line = line
                break

        year_columns = {}
        if header_line:
            years = re.findall(r"20x\d+", header_line.lower())
            for idx, y in enumerate(years):
                year_columns[y] = idx

        # 4️⃣ find row containing requested metric
        for line in lines:
            if requested_metric in line.lower():

                numbers = re.findall(r"\d[\d,\.]*", line)

                if not numbers:
                    continue

                # If year is specified and exists in header
                if requested_year and requested_year in year_columns:
                    col_index = year_columns[requested_year]

                    if col_index < len(numbers):
                        return numbers[col_index]

                # fallback → return last numeric value
                return numbers[-1]

    return None
    """

def extract_exact_numeric(context_docs, question):

    q_lower = question.lower()

    # Expanded metric matching
    metric_aliases = {
        "net sales": ["net sales", "sales"],
        "revenue": ["revenue", "net sales", "turnover"],
        "interest": ["interest", "finance cost", "financial cost", "interest expense"],
        "profit": ["profit", "pbt", "pat"],
        "tax": ["tax"],
        "dividend": ["dividend"],
        "retained earnings": ["retained earnings"],
        "cost of goods sold": ["cost of goods sold", "cogs"]
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

    # Detect year
    year_match = re.search(r"20x\d+", q_lower)
    requested_year = year_match.group() if year_match else None

    # Combine all context (important if table split across chunks)
    full_text = "\n".join(context_docs)
    lines = [l.strip() for l in full_text.split("\n") if l.strip()]

    header_line = None
    year_columns = {}

    # Find header row containing years
    for line in lines:
        years_found = re.findall(r"20x\d+", line.lower())
        if years_found:
            header_line = line
            for idx, y in enumerate(years_found):
                year_columns[y] = idx
            break

    # Find metric row
    for line in lines:
        for alias in metric_aliases[requested_metric]:
            if alias in line.lower():

                numbers = re.findall(r"\d[\d,\.]*", line)
                if not numbers:
                    continue

                # If specific year requested
                if requested_year and requested_year in year_columns:
                    col_index = year_columns[requested_year]

                    if col_index < len(numbers):
                        return numbers[col_index]

                # fallback → return last number
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

Using ONLY the context below, explain clearly in a comparison table.

Include:
- Definition
- Formula
- Key differences
- Example

Context:
{context}

Question:
{question}

Answer in table format:
"""

    return f"""
You are a finance professor.

Using ONLY the context below, give a detailed explanation.

Rules:
- Minimum 6 sentences
- Explain concept clearly
- Include formula if applicable
- Include simple example
- No repetition

Context:
{context}

Question:
{question}

Detailed Answer:
"""


# ==============================
# MAIN RAG FUNCTION
# ==============================

def rag_answer(question: str):

    load_models()
    load_vector_store()

    context_docs, scores = retrieve(question)
    max_score = max(scores) if scores else 0

    # STRICT NUMERIC MODE
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

        advice_prompt = f"""
You are a certified financial advisor.

Give 5 practical bullet points for a student investor.
Low risk focused. No repetition.

Question:
{question}

Answer:
"""

        result = _llm(advice_prompt)
        answer = remove_repetition(result[0]["generated_text"].strip())

        return {
            "question": question,
            "answer": answer,
            "mode": "LLM-Advice"
        }

    # GENERAL MODE
    if max_score < 0.30:

        general_prompt = f"""
Explain clearly in detail.
Include formula if applicable and simple example.
Minimum 6 sentences.

Question:
{question}

Answer:
"""

        result = _llm(general_prompt)
        answer = remove_repetition(result[0]["generated_text"].strip())

        return {
            "question": question,
            "answer": answer,
            "mode": "LLM-General"
        }

    # RAG MODE
    if not context_docs:
        return {
            "question": question,
            "answer": "Answer not found in document",
            "mode": "Not-Found"
        }

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
