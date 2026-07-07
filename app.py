import os
import math
import urllib.request
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

SCORE_THRESHOLD = 0.80
NMS_THRESHOLD = 0.30
TOP_K = 5000

"""
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"

MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

def download_model_if_needed() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        return

    print(f"Downloading YuNet model to {MODEL_PATH}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


download_model_if_needed()
"""

MODEL_PATH = Path("models/face_detection_yunet_2023mar.onnx")

app = FastAPI(title="YuNet Live Face Detection API")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


def decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Invalid image")

    return frame


def detect_faces(frame: np.ndarray):
    h, w = frame.shape[:2]

    detector = cv2.FaceDetectorYN.create(
        model=str(MODEL_PATH),
        config="",
        input_size=(w, h),
        score_threshold=SCORE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        top_k=TOP_K,
    )

    _, faces = detector.detect(frame)

    detections = []

    if faces is not None:
        for face in faces:
            x, y, bw, bh = face[:4]
            score = face[-1]

            detections.append({
                "x": int(x),
                "y": int(y),
                "w": int(bw),
                "h": int(bh),
                "score": float(score),
            })

    return detections

@app.post("/api/frame")
async def analyze_frame(frame: UploadFile = File(...)):
    try:
        image_bytes = await frame.read()
        image = decode_image(image_bytes)
        detections = detect_faces(image)

        n_faces = len(detections)

        if n_faces == 0:
            status = "FACE_NOT_FOUND"
        elif n_faces == 1:
            status = "SINGLE_FACE"
        else:
            status = "MULTIPLE_FACES"

        return {
            "faces": n_faces,
            "status": status,
            "detections": detections,
        }

    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": str(exc),
                "faces": 0,
                "status": "FACE_NOT_FOUND",
                "detections": [],
            },
        )


@app.get("/health")
def health():
    return {"status": "ok", "model": "YuNet"}


breakpoint