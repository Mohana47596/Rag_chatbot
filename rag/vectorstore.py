import os
import pickle
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

PDF_PATH = "uploaded_files/your_pdf.pdf"  # <-- change to your PDF path
SAVE_PATH = "rag/vector_store.pkl"

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def main():
    print("Loading embedding model...")
    model = SentenceTransformer("all-mpnet-base-v2")

    print("Extracting PDF text...")
    text = extract_text_from_pdf(PDF_PATH)

    print("Chunking text...")
    chunks = chunk_text(text)

    print("Generating embeddings...")
    embeddings = model.encode(chunks)

    print("Saving vector store...")
    os.makedirs("rag", exist_ok=True)

    with open(SAVE_PATH, "wb") as f:
        pickle.dump(list(zip(chunks, embeddings)), f)

    print("✅ Vector store created successfully!")

if __name__ == "__main__":
    main()
