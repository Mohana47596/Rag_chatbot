from fastapi import APIRouter
from pydantic import BaseModel
from rag.pipeline import rag_answer
import traceback

router = APIRouter()

class ChatRequest(BaseModel):
    question: str

@router.post("/chat")
def chat(req: ChatRequest):
    print("Incoming Question:", req.question)

    try:
        result = rag_answer(req.question)
        print("Answer:", result)
        return result

    except Exception as e:
        print("Error occurred:")
        traceback.print_exc()

        return {
            "answer": "Internal server error occurred",
            "error": str(e)
        }
