import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Body, Query, Depends, status, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any, Union
import os
import sys
import shutil
import uuid
import requests
from datetime import datetime
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Import routers
print("Importing routers...")
try:
    from app.api.parallel_detection import router as parallel_router
    print("✅ Successfully imported parallel detection router")
    
    from app.api.sequential_detection import router as sequential_router
    print("✅ Successfully imported sequential detection router")
    
    from app.api.unified_detection import router as unified_router
    print("✅ Successfully imported unified detection router")
    
except ImportError as e:
    print(f"❌ Error importing routers: {e}")
    raise

# Initialize FastAPI app
print("\n=== Initializing FastAPI app ===")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

app = FastAPI(title="Deepfake Detection API")
print("FastAPI app initialized")

# Include the routers
print("\n=== Including API routers ===")

# Include unified detection router
try:
    print("\nIncluding unified detection router...")
    app.include_router(
        unified_router,
        prefix="/api/unified",
        tags=["Unified Detection"],
        responses={404: {"description": "Not found"}},
    )
    print("✅ Unified detection router included")

except Exception as e:
    print(f"❌ Error including unified detection router: {e}")
    raise

# Include parallel detection router
try:
    # Debug: Print router details before inclusion
    print("\nParallel detection router details:")
    print(f"  Router prefix: {getattr(parallel_router, 'prefix', 'Not set')}")
    print(f"  Router tags: {getattr(parallel_router, 'tags', 'Not set')}")
    print(f"  Router routes: {[route.path for route in parallel_router.routes]}")
    
    # Include the router with the prefix and tags
    app.include_router(
        parallel_router,
        prefix="/api/parallel",
        tags=["Parallel Detection"]
    )
    print("✅ Successfully included parallel detection router at /api/parallel")
    
except Exception as e:
    print(f"❌ Error including parallel detection router: {e}")
    raise

# Include sequential detection router
try:
    # Debug: Print router details before inclusion
    print("\nSequential detection router details:")
    print(f"  Router prefix: {getattr(sequential_router, 'prefix', 'Not set')}")
    print(f"  Router tags: {getattr(sequential_router, 'tags', 'Not set')}")
    print(f"  Router routes: {[route.path for route in sequential_router.routes]}")
    
    # Include the router with the prefix and tags
    app.include_router(
        sequential_router,
        prefix="/api/sequential",
        tags=["Sequential Detection"]
    )
    print("✅ Successfully included sequential detection router at /api/sequential")
    
    # Verify all routers were included
    print("\nRegistered routes after including all routers:")
    for route in app.routes:
        print(f"- {route.path} ({', '.join(getattr(route, 'tags', ['No tags']))})")
        
except Exception as e:
    print(f"❌ Error including sequential detection router: {e}")
    print(f"Error type: {type(e).__name__}")
    print(f"Error details: {str(e)}")
    import traceback
    traceback.print_exc()
    raise

# Optional imports with fallback
FACE_INDEXING_AVAILABLE = False

try:
    from app.api.face_indexing import router as face_router
    from app.api.youtube_webhook import router as youtube_router
    from app.api.videos import router as videos_router
    from app.schemas.video import VideoResponse
    FACE_INDEXING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Face indexing features disabled - {str(e)}")
    FACE_INDEXING_AVAILABLE = False

# Include optional routers if dependencies are available
if FACE_INDEXING_AVAILABLE:
    app.include_router(face_router, prefix="/api/face", tags=["face-indexing"])
    app.include_router(youtube_router, prefix="/api/youtube", tags=["youtube"])
    app.include_router(videos_router, prefix="/api/videos", tags=["videos"])

    # Create a separate router for video-related endpoints
    from fastapi import APIRouter
    video_router = APIRouter(prefix="/api/videos", tags=["videos"])

    @video_router.post("/index_face")
    async def index_face_endpoint(file: UploadFile = File(...)):
        """
        Endpoint to upload and index a new face video.
        """
        try:
            # Save uploaded file temporarily
            temp_path = f"temp_uploads/{uuid.uuid4()}_{file.filename}"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Process and index the face
            result = await index_face(temp_path)
            
            # Clean up temp file
            os.remove(temp_path)
            
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @video_router.post("/yt_webhook")
    async def youtube_webhook_endpoint(
        video_url: str = Query(None, description="YouTube video URL"),
        body: dict = Body(None, description="Request body containing video URL")
    ):
        """
        Endpoint to process a new YouTube video for deepfake detection.
        Accepts video URL either as a query parameter or in the request body.
        """
        try:
            # Get video URL from either query parameter or request body
            url = video_url or (body.get("video_url") if body else None)
            
            if not url:
                raise HTTPException(
                    status_code=400,
                    detail="Video URL must be provided either as a query parameter or in the request body"
                )
                
            # Log the received URL for debugging
            print(f"Received URL: {url}")
            
            # Process the video
            result = await process_youtube_video(url)
            
            # Check if the result indicates an error
            if result.get("status") == "error":
                raise HTTPException(
                    status_code=400,
                    detail=result.get("message", "Unknown error occurred while processing video")
                )
                
            return result
        except HTTPException:
            raise
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )

    @video_router.get("/{user_id}", response_model=List[VideoResponse])
    async def get_videos_endpoint(user_id: str):
        """
        Endpoint to get all videos related to a specific user.
        """
        try:
            videos = await get_user_videos(user_id)
            return videos
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    app.include_router(video_router)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "https://deepfake-*.vercel.app",
        "https://deepfake-6fjgqqqpx-prajwals-projects-3333b665.vercel.app",
        "https://deepfake-7idhjm083-prajwals-projects-3333b665.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Add CORS headers to all responses (complementary to CORSMiddleware)
@app.middleware("http")
async def add_cors_headers(request, call_next):
    if request.method == "OPTIONS":
        return JSONResponse(status_code=200, headers={
            "Access-Control-Allow-Origin": ", ".join([
                "http://localhost:3000",
                "https://deepfake-*.vercel.app",
                "https://deepfake-6fjgqqqpx-prajwals-projects-3333b665.vercel.app",
                "https://deepfake-7idhjm083-prajwals-projects-3333b665.vercel.app",
            ]),
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true"
        })
    
    response = await call_next(request)
    return response

# Create necessary directories
os.makedirs("temp_uploads", exist_ok=True)
os.makedirs("processed_videos", exist_ok=True)

# Modal endpoint configuration
MODAL_ENDPOINT = "https://ds21ai038--deepfake-ensemble-detector-fastapi-app.modal.run"

# Mock database for face indices (replace with actual database in production)
face_indices_db = []

# Pydantic models for face indexing
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

# Modal client endpoints
@app.post("/api/detect")
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
        
        response = requests.post(MODAL_ENDPOINT, files=files, data=data, timeout=300)
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calling Modal API: {str(e)}"
        )

@app.post("/api/face_indices", response_model=FaceIndexResponse)
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

@app.get("/api/face_indices")
async def list_face_indices(user_id: str):
    """
    List all face indices for a user
    """
    try:
        # In a real implementation, filter by user_id from database
        user_faces = [face for face in face_indices_db if face.get("user_id") == user_id]
        return user_faces
    except Exception as e:
        logger.error(f"Error listing face indices: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve face indices"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    
    # Configure Uvicorn logging
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelprefix)s %(client_addr)s - \"%(request_line)s\" %(status_code)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelprefix)s %(message)s"
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug",
        log_config=log_config,
        access_log=True
    )