"""
FastAPI endpoint for sequential deepfake detection with weighted decisions.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import logging
import os
import shutil
import tempfile
import time
from typing import List, Dict, Any

# Import our sequential processor
from models.sequential_processor import SequentialProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the router without prefix or tags - they'll be set in main.py
router = APIRouter()

# Initialize the processor (lazy load on first request)
_processor = None

def get_processor():
    """Get or initialize the sequential processor."""
    global _processor
    if _processor is None:
        logger.info("Initializing sequential processor...")
        _processor = SequentialProcessor()
    return _processor

@router.post("/detect/image", response_model=Dict[str, Any])
async def detect_image(file: UploadFile = File(...), confidence_threshold: float = 0.5):
    """
    Process an image through the sequential detection pipeline.
    
    Args:
        file: Uploaded image file
        confidence_threshold: Threshold for binary classification (default: 0.5)
        
    Returns:
        JSON with detection results including individual model outputs and final decision
    """
    try:
        # Validate confidence threshold
        if not 0 < confidence_threshold < 1:
            raise HTTPException(
                status_code=400,
                detail="Confidence threshold must be between 0 and 1"
            )
        
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image"
            )
        
        # Get processor and process image
        processor = get_processor()
        start_time = time.time()
        
        # Process the frame
        result = processor.process_frame(image)
        
        processing_time = time.time() - start_time
        
        # Format response
        response = {
            "status": "success",
            "processing_time_seconds": processing_time,
            "final_decision": "fake" if result["is_deepfake"] else "real",
            "confidence": result["confidence"],
            "score": result["final_decision"],
            "model_predictions": result["predictions"],
            "metadata": {
                "filename": file.filename,
                "resolution": f"{image.shape[1]}x{image.shape[0]}",
                "channels": image.shape[2] if len(image.shape) > 2 else 1,
                "models_used": [pred["model"] for pred in result["predictions"]]
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@router.post("/detect/video", response_model=Dict[str, Any])
async def detect_video(
    file: UploadFile = File(...),
    frame_skip: int = 10,
    max_frames: int = 100,
    confidence_threshold: float = 0.5
):
    """
    Process a video file through the sequential detection pipeline.
    
    Args:
        file: Video file to process (MP4, AVI, MOV, MKV)
        frame_skip: Number of frames to skip between processing (default: 10)
        max_frames: Maximum number of frames to process (default: 100)
        confidence_threshold: Threshold for binary classification (default: 0.5)
        
    Returns:
        Dict containing detection results and metadata
    """
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, file.filename)
    
    try:
        # Validate parameters
        if frame_skip < 0:
            raise HTTPException(
                status_code=400,
                detail="frame_skip must be >= 0"
            )
            
        if max_frames <= 0:
            raise HTTPException(
                status_code=400,
                detail="max_frames must be > 0"
            )
            
        if not 0 < confidence_threshold < 1:
            raise HTTPException(
                status_code=400,
                detail="confidence_threshold must be between 0 and 1"
            )
        
        # Save uploaded file
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Open video file
        cap = cv2.VideoCapture(temp_file)
        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Could not open video file"
            )
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Initialize variables
        frame_count = 0
        processed_frames = 0
        total_score = 0.0
        frame_results = []
        processor = get_processor()
        
        start_time = time.time()
        
        # Process video frames
        while True:
            ret, frame = cap.read()
            if not ret or processed_frames >= max_frames:
                break
                
            # Skip frames according to frame_skip
            if frame_count % (frame_skip + 1) != 0:
                frame_count += 1
                continue
            
            try:
                # Process frame
                result = processor.process_frame(frame)
                
                # Store frame result
                frame_result = {
                    "frame_number": frame_count,
                    "timestamp": frame_count / fps if fps > 0 else 0,
                    "is_deepfake": result["is_deepfake"],
                    "confidence": result["confidence"],
                    "score": result["final_decision"],
                    "models_used": [pred["model"] for pred in result["predictions"]]
                }
                frame_results.append(frame_result)
                
                # Update running total
                total_score += result["final_decision"]
                processed_frames += 1
                
            except Exception as e:
                logger.warning(f"Error processing frame {frame_count}: {str(e)}")
            
            frame_count += 1
        
        # Calculate final results
        if processed_frames == 0:
            raise HTTPException(
                status_code=400,
                detail="No frames were processed"
            )
        
        processing_time = time.time() - start_time
        avg_score = total_score / processed_frames
        is_deepfake = avg_score >= confidence_threshold
        
        # Prepare response
        response = {
            "status": "success",
            "metadata": {
                "filename": file.filename,
                "duration_seconds": duration,
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "frame_skip": frame_skip,
                "resolution": f"{width}x{height}",
                "fps": fps
            },
            "processing_stats": {
                "total_time_seconds": processing_time,
                "frames_processed_per_second": processed_frames / processing_time if processing_time > 0 else 0,
                "processing_fps": fps * (frame_skip + 1) if fps > 0 else 0
            },
            "results": {
                "overall_prediction": "fake" if is_deepfake else "real",
                "overall_confidence": abs(avg_score - 0.5) * 2,  # Convert to 0-1 confidence
                "average_score": avg_score,
                "frame_level_results": frame_results[:10]  # Return first 10 frames for inspection
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )
        
    finally:
        # Clean up
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

@router.get("/models")
async def list_models():
    """
    List all available models and their status.
    """
    try:
        processor = get_processor()
        available_models = processor.available_models
        
        response = {
            "status": "success",
            "available_models": {
                model_id: {
                    "weight": weight,
                    "status": "loaded"
                }
                for model_id, weight in available_models.items()
            },
            "total_models": len(available_models),
            "weights_sum": sum(available_models.values())
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error listing models: {str(e)}"
        )

# Export the router
__all__ = ["router"]

# This ensures the router is properly imported when the module is loaded
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(router, host="0.0.0.0", port=8000)
