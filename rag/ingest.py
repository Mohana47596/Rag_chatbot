# backend/rag/ingest.py

import os
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

VECTOR_PATH = "rag/vector_store/chunks.pkl"

def ingest_pdf(file_path: str):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    if not text.strip():
        raise ValueError("PDF has no readable text")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    os.makedirs(os.path.dirname(VECTOR_PATH), exist_ok=True)

    with open(VECTOR_PATH, "wb") as f:
        pickle.dump(chunks, f)

    return {"chunks": len(chunks)}
