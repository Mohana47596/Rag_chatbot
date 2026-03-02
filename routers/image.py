from fastapi import APIRouter, UploadFile, File
from transformers import pipeline
from PIL import Image
import io

router = APIRouter()

vision_model = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base"
)

@router.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    result = vision_model(image)

    return {
        "description": result[0]["generated_text"]
    }