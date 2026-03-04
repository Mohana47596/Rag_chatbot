import os
import pickle
from fastapi import APIRouter, UploadFile, File
from sentence_transformers import SentenceTransformer
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

router = APIRouter()

UPLOAD_FOLDER = "uploads"
VECTOR_PATH = "rag/vector_store.pkl"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("rag", exist_ok=True)


# ==============================
# PDF TEXT EXTRACTION
# ==============================

def extract_text_from_pdf(pdf_path):
    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    return full_text


# ==============================
# TEXT CHUNKING
# ==============================

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)


# ==============================
# UPLOAD ROUTE
# ==============================

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    print("Extracting text from PDF...")
    text = extract_text_from_pdf(file_path)

    print("Chunking text...")
    chunks = chunk_text(text)

    print("Creating embeddings...")
    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(chunks)

    data = {
        "documents": chunks,
        "embeddings": embeddings
    }

    with open(VECTOR_PATH, "wb") as f:
        pickle.dump(data, f)

    print("Vector store created successfully.")

    return {"message": "PDF uploaded and vector store created successfully."}
