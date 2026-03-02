"""
from fastapi import APIRouter
from pydantic import BaseModel
from rag.pipeline import rag_answer
import traceback

router = APIRouter()

class ChatRequest(BaseModel):
    question: str


@router.post("/chat")
def chat(req: ChatRequest):
    print("---- Incoming Question ----")
    print(req.question)

    try:
        result = rag_answer(req.question)
        print("---- Answer Generated ----")
        print(result)
        return result

    except Exception as e:
        print("---- ERROR OCCURRED ----")
        traceback.print_exc()
        return {
            "answer": "Internal server error occurred",
            "used_tables": False,
            "error": str(e)
        }
"""
from fastapi import APIRouter
from pydantic import BaseModel
from rag.pipeline import rag_answer

router = APIRouter()

class Question(BaseModel):
    question: str



@router.post("/chat")
def chat(data: Question):
    try:
        result = rag_answer(data.question)
        return result
    except Exception as e:
        return {
            "answer": "Error generating response",
            "error": str(e)
        }