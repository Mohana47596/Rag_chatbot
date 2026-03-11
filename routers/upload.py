import os
import pickle
from fastapi import APIRouter, UploadFile, File
from sentence_transformers import SentenceTransformer
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

router = APIRouter()

UPLOAD_FOLDER = "uploads"
VECTOR_PATH = "rag/vector_store.pkl"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("rag", exist_ok=True)

# Load model only once (important)
model = SentenceTransformer("all-MiniLM-L6-v2")

# ==============================
# PDF TEXT EXTRACTION
# ==============================

def extract_text_from_pdf(pdf_path):
    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:20]:  # limit pages to avoid memory crash
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    return full_text


# ==============================
# TEXT CHUNKING
# ==============================

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
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

    print("Extracting text...")
    text = extract_text_from_pdf(file_path)

    print("Chunking...")
    chunks = chunk_text(text)

    print("Creating embeddings...")
    embeddings = model.encode(chunks)

    data = {
        "documents": chunks,
        "embeddings": embeddings
    }

    with open(VECTOR_PATH, "wb") as f:
        pickle.dump(data, f)

    print("Vector store saved.")

    return {"message": "PDF uploaded and vector store created successfully."}
