"""
Deepfake Detection API Client for Modal Endpoint with Face Indexing
"""
import os
import cv2
import requests
import tempfile
import numpy as np
import uuid
import logging
import ssl
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, APIRouter, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Deepfake Detection API with Face Indexing")
router = APIRouter()

# CORS middleware with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://localhost:3000"
    ],
    allow_credentials=False,  # Disable credentials to work with frontend
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Production Modal endpoint - ensemble_detector.py deployed
MODAL_ENDPOINT = "https://ds21ai038--deepfake-ensemble-detector-detect.modal.run"

# Mock database for face indices (replace with actual database in production)
face_indices_db = []

# Pydantic models
class FaceIndex(BaseModel):
    id: str
    user_id: str
    name: str
    created_at: str
    vector_embedding: Optional[List[float]] = None

class FaceIndexResponse(BaseModel):
    status: str
    face_index: Optional[FaceIndex] = None
    message: Optional[str] = None

@router.post("/detect")
async def detect(
    file: UploadFile = File(...),
    video_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    video_url: Optional[str] = Form(None),
    include_details: bool = Form(False)
) -> Dict[str, Any]:
    """Forward detection request to Modal API"""
    try:
        file_bytes = await file.read()
        files = {"file": (file.filename, file_bytes, file.content_type)}
        data = {
            "video_id": video_id or "",
            "user_id": user_id or "",
            "video_url": video_url or "",
            "include_details": str(include_details).lower()
        }
        
        # Add SSL verification and retry logic
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        session = requests.Session()
        session.verify = False  # Disable SSL verification for now
        
        response = session.post(
            MODAL_ENDPOINT, 
            files=files, 
            data=data, 
            timeout=300,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.SSLError as e:
        logger.error(f"SSL Error with Modal API: {str(e)}")
        # Try with a different approach
        try:
            import ssl
            import urllib3
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            # Create a session with custom SSL context
            session = requests.Session()
            
            # Disable SSL warnings
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            # Retry strategy
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            response = session.post(
                MODAL_ENDPOINT,
                files=files,
                data=data,
                timeout=300,
                verify=False,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json',
                    'Connection': 'keep-alive'
                }
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as retry_error:
            logger.error(f"Retry failed: {str(retry_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Modal API connection failed. SSL Error: {str(e)}. Retry Error: {str(retry_error)}"
            )
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error calling Modal API: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy", "service": "modal-client"}

@router.get("/test-modal")
async def test_modal_connection():
    """Test Modal API connection"""
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        session = requests.Session()
        session.verify = False
        
        # Simple GET request to test connection
        response = session.get(
            "https://ds21ai038--deepfake-ensemble-detector-fastapi-app.modal.run/health",
            timeout=30,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        
        return {
            "status": "success",
            "modal_status": response.status_code,
            "modal_response": response.text[:200] if response.text else "No response body"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Modal API connection test failed"
        }

@router.post("/face_indices", response_model=FaceIndexResponse)
async def create_face_index(
    user_id: str = Form(...),
    name: str = Form("Untitled Face"),
    file: UploadFile = File(...)
):
    """
    Create a new face index from an uploaded image/video
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # In a real implementation, you would:
        # 1. Process the file to extract face embeddings
        # 2. Store the embeddings in a vector database
        # 3. Return the face index ID
        
        # For now, we'll create a mock response
        face_id = str(uuid.uuid4())
        new_face = {
            "id": face_id,
            "user_id": user_id,
            "name": name,
            "created_at": datetime.utcnow().isoformat(),
            "vector_embedding": [0.1, 0.2, 0.3]  # Mock embedding
        }
        
        face_indices_db.append(new_face)
        
        return {
            "status": "success",
            "face_index": new_face,
            "message": "Face index created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating face index: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create face index: {str(e)}"
        )

@router.get("/face_indices", response_model=List[FaceIndex])
async def list_face_indices(user_id: str):
    """
    List all face indices for a user
    """
    try:
        # In a real implementation, filter by user_id from database
        user_faces = [face for face in face_indices_db if face["user_id"] == user_id]
        return user_faces
    except Exception as e:
        logger.error(f"Error listing face indices: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve face indices"
        )

@router.delete("/face_indices/{face_id}")
async def delete_face_index(face_id: str):
    """
    Delete a face index
    """
    try:
        global face_indices_db
        original_length = len(face_indices_db)
        face_indices_db = [face for face in face_indices_db if face["id"] != face_id]
        
        if len(face_indices_db) == original_length:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Face index not found"
            )
        
        return {"status": "success", "message": "Face index deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting face index: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete face index"
        )

# Include the router with API prefix
app.include_router(router, prefix="/api")

# Add CORS headers to all responses
@app.middleware("http")
async def add_cors_headers(request, call_next):
    # Define allowed origins
    allowed_origins = [
        "http://localhost:3000",  # Development
        "https://localhost:3000",  # Development HTTPS
        # Add your Vercel domain here when you get it
        # "https://your-app-name.vercel.app"
    ]
    
    # Get the origin from the request
    origin = request.headers.get("origin")
    
    # Check if origin is allowed or if it's a vercel.app domain
    allowed_origin = None
    if origin:
        if origin in allowed_origins or origin.endswith(".vercel.app"):
            allowed_origin = origin
    
    # Handle preflight requests
    if request.method == "OPTIONS":
        response = JSONResponse(content={}, status_code=200)
        if allowed_origin:
            response.headers["Access-Control-Allow-Origin"] = allowed_origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Max-Age"] = "86400"
        return response
    
    response = await call_next(request)
    if allowed_origin:
        response.headers["Access-Control-Allow-Origin"] = allowed_origin
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

if __name__ == "__main__":
    uvicorn.run("modal_client:app", host="0.0.0.0", port=8000, reload=True)

