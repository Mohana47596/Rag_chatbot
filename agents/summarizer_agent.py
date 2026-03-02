# backend/agents/summarizer_agent.py

from transformers import pipeline

summarizer_pipeline = None

def summarizer_agent(text: str) -> str:
    """
    Takes LLM output and summarizes / cleans it
    """
    global summarizer_pipeline
    if not text or len(text.strip()) == 0:
        return ""

    # Lazy load summarizer
    if summarizer_pipeline is None:
        summarizer_pipeline = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )

    result = summarizer_pipeline(
        text, max_length=200, min_length=50, do_sample=False
    )
    return result[0]["summary_text"]
