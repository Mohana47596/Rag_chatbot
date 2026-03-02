from fastapi import APIRouter, UploadFile, File
import whisper
import tempfile

router = APIRouter()

model = whisper.load_model("base")

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    result = model.transcribe(tmp_path)

    return {
        "text": result["text"]
    }