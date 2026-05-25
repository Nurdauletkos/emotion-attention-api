"""
FastAPI сервер - Эмоция + Назар бақылау
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import shutil
import os
import tempfile

from emotion_detector import EmotionDetector
from attention_detector import AttentionDetector

app = FastAPI(
    title="EmotionAI API",
    description="Эмоция анықтау + Назар бақылау AI жүйесі",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Детекторларды бір рет жүктеу
emotion_detector = EmotionDetector(detector_backend='opencv')
attention_detector = AttentionDetector(detector_backend='opencv')


class Base64ImageRequest(BaseModel):
    image: str


# ============ БАСТЫ БЕТ ============

@app.get("/")
async def root():
    return {
        "service": "EmotionAI API",
        "version": "2.0.0",
        "status": "жұмыс істеп тұр",
        "features": ["emotion_detection", "attention_monitoring"]
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# ============ ЭМОЦИЯ ============

@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Файл сурет болуы керек")
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Суретті оқу мүмкін болмады")
        results = emotion_detector.detect_from_image(img)
        return {"success": True, "faces_count": len(results), "faces": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Талдау қатесі: {str(e)}")


@app.post("/analyze/base64")
async def analyze_base64(request: Base64ImageRequest):
    try:
        result = emotion_detector.detect_from_base64(request.image)
        if not result['success']:
            return JSONResponse(status_code=400, content=result)
        return result
    except Exception as e:
        raise HTTPException(500, f"Талдау қатесі: {str(e)}")


@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...), sample_rate: int = 30):
    if not file.content_type.startswith('video/'):
        raise HTTPException(400, "Файл видео болуы керек")
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename)[1]
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        result = emotion_detector.detect_from_video(tmp_path, sample_rate)
        return result
    except Exception as e:
        raise HTTPException(500, f"Видео талдау қатесі: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============ НАЗАР БАҚЫЛАУ ============

@app.post("/attention/base64")
async def attention_base64(request: Base64ImageRequest):
    """Камера кадрынан назар жағдайын анықтау"""
    try:
        result = attention_detector.detect_from_base64(request.image)
        return result
    except Exception as e:
        raise HTTPException(500, f"Назар талдау қатесі: {str(e)}")


@app.post("/attention/image")
async def attention_image(file: UploadFile = File(...)):
    """Суреттен назар жағдайын анықтау"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Файл сурет болуы керек")
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Суретті оқу мүмкін болмады")
        result = attention_detector.detect_from_image(img)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Назар талдау қатесі: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
