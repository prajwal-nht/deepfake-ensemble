"""
Modal deployment of the Deepfake Detection API
"""
import modal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, APIRouter, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pydantic import BaseModel
import requests
import tempfile
import uuid
import logging
import ssl
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Create Modal app
app = modal.App("deepfake-detection-api")

# Create FastAPI app
web_app = FastAPI(title="Deepfake Detection API with Face Indexing")
router = APIRouter()

# CORS middleware - allow all origins for Vercel deployment
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

MODAL_ENDPOINT = "https://ds21ai038--deepfake-ensemble-detector-fastapi-app.modal.run/detect"

# Mock database for face indices
face_indices_db = []

# Pydantic models
class FaceIndex(BaseModel):
    id: str
    user_id: str
    face_encoding: List[float]
    image_path: str
    created_at: datetime

class DetectionResult(BaseModel):
    video_id: str
    is_deepfake: bool
    confidence: float
    processed_at: str
    face_matches: Optional[List[Dict[str, Any]]] = []
    frame_count: Optional[int] = None
    processed_frames: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    message: str

# Health check endpoint
@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", message="Deepfake Detection API is running")

# Detection endpoint
@router.post("/detect", response_model=DetectionResult)
async def detect_deepfake(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Prepare files for Modal endpoint
        with open(temp_file_path, 'rb') as f:
            files = {'file': (file.filename, f, file.content_type)}
            
            # Configure session with SSL and retry
            session = requests.Session()
            session.verify = False  # Disable SSL verification
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # Make request to Modal endpoint
            response = session.post(
                MODAL_ENDPOINT,
                files=files,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                return DetectionResult(
                    video_id=str(uuid.uuid4()),
                    is_deepfake=result.get('is_deepfake', False),
                    confidence=result.get('confidence', 0.0),
                    processed_at=datetime.now().isoformat(),
                    face_matches=result.get('face_matches', []),
                    frame_count=result.get('frame_count'),
                    processed_frames=result.get('processed_frames')
                )
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Modal endpoint error: {response.text}"
                )
                
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )

# Face indexing endpoints (mock implementation)
@router.post("/face_indices")
async def create_face_index(user_id: str = Form(...), file: UploadFile = File(...)):
    face_id = str(uuid.uuid4())
    face_index = {
        "id": face_id,
        "user_id": user_id,
        "filename": file.filename,
        "created_at": datetime.now().isoformat()
    }
    face_indices_db.append(face_index)
    return {"id": face_id, "message": "Face indexed successfully"}

@router.get("/face_indices")
async def get_face_indices(user_id: str):
    user_faces = [face for face in face_indices_db if face["user_id"] == user_id]
    return {"faces": user_faces}

@router.delete("/face_indices/{face_id}")
async def delete_face_index(face_id: str):
    global face_indices_db
    face_indices_db = [face for face in face_indices_db if face["id"] != face_id]
    return {"message": "Face index deleted successfully"}

# Include router
web_app.include_router(router, prefix="/api")

# Add CORS headers to all responses
@web_app.middleware("http")
async def add_cors_headers(request, call_next):
    # Allow all origins for Vercel deployment
    if request.method == "OPTIONS":
        response = JSONResponse(content={}, status_code=200)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Max-Age"] = "86400"
        return response
    
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Modal deployment
@app.function(
    image=modal.Image.debian_slim().pip_install([
        "fastapi",
        "uvicorn",
        "python-multipart",
        "requests",
        "opencv-python-headless",
        "numpy",
        "urllib3"
    ]),
    allow_concurrent_inputs=10,
)
@modal.asgi_app()
def fastapi_app():
    return web_app