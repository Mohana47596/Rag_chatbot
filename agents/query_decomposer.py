# backend/agents/query_decomposer.py

def decompose_query(question: str):
    q = question.lower()

    if "compare" in q or "difference" in q:
        intent = "compare"
    elif "explain" in q or "describe" in q:
        intent = "explain"
    elif "define" in q or "what is" in q:
        intent = "define"
    else:
        intent = "general"

    return {
        "intent": intent,
        "original_question": question
    }
