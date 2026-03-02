from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.chat import router as chat_router
from routers.upload import router as upload_router
from routers.image import router as image_router
from routers.audio import router as audio_router

app = FastAPI()

# ✅ ADD THIS CORS SECTION
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # allow all (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(upload_router)
app.include_router(chat_router)
app.include_router(image_router)
app.include_router(audio_router)