import os
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image
import io

from pipeline.demographic_analyzer import DemographicAnalyzer, DemographicResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="",  # Remove prefix here since we'll add it in main_updated.py
    tags=["Demographic Analysis"],  # More descriptive tag
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    },
)

# Initialize the analyzer
analyzer = DemographicAnalyzer()

async def process_image_file(file: UploadFile) -> np.ndarray:
    """Process uploaded image file to numpy array."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        return np.array(image)
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid image file")

@router.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    min_quality: float = 0.1,  # Lowered default to ensure we see all detections
    max_faces: Optional[int] = None
):
    """
    Analyze an image for demographic information.
    
    Args:
        file: Image file to analyze
        min_quality: Minimum quality score (0-1) for face detection
        max_faces: Maximum number of faces to process (None for all)
        
    Returns:
        List of demographic results for each detected face
    """
    try:
        logger.info(f"[API] Starting analysis for image: {file.filename}")
        logger.info(f"[API] Parameters - min_quality: {min_quality}, max_faces: {max_faces}")
        
        # Process the image
        logger.info("[API] Processing image file...")
        image = await process_image_file(file)
        logger.info(f"[API] Image processed. Shape: {image.shape}, Type: {image.dtype}")
        
        # Save the uploaded image for debugging
        debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        input_path = os.path.join(debug_dir, 'api_input_image.jpg')
        cv2.imwrite(input_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        logger.info(f"[API] Saved input image to: {os.path.abspath(input_path)}")
        
        # Analyze the image
        logger.info("[API] Starting demographic analysis...")
        results = analyzer.analyze(image)
        logger.info(f"[API] Analysis complete. Raw results type: {type(results)}")
        
        if results is None:
            logger.warning("[API] Analyzer returned None")
            return JSONResponse(content={"error": "Analysis failed", "results": []})
            
        logger.info(f"[API] Raw results length: {len(results) if hasattr(results, '__len__') else 'N/A'}")
        
        if hasattr(results, '__len__') and len(results) > 0:
            logger.info(f"[API] First result type: {type(results[0])}")
            logger.info(f"[API] First result keys: {dir(results[0]) if hasattr(results[0], '__dict__') else 'N/A'}")
        
        # Filter results by quality
        filtered_results = []
        for i, result in enumerate(results):
            try:
                quality = getattr(result, 'quality_score', 0)
                logger.info(f"[API] Face {i} - Quality: {quality:.2f}, BBox: {getattr(result, 'bbox', 'N/A')}")
                if quality >= min_quality:
                    filtered_results.append(result)
            except Exception as e:
                logger.error(f"[API] Error processing face {i}: {str(e)}", exc_info=True)
        
        logger.info(f"[API] Found {len(filtered_results)} faces after quality filtering")
        
        # Limit number of faces if specified
        if max_faces is not None and max_faces > 0:
            filtered_results = filtered_results[:max_faces]
            logger.info(f"[API] Limited to {len(filtered_results)} faces based on max_faces parameter")
        
        # Convert results to dict for JSON serialization
        response = []
        for i, result in enumerate(filtered_results):
            try:
                face_data = {
                    "face_id": i,
                    "bbox": getattr(result, 'bbox', []),
                    "age": getattr(result, 'age', -1),
                    "gender": getattr(result, 'gender', 'unknown'),
                    "gender_confidence": float(getattr(result, 'gender_confidence', 0.0)),
                    "race": getattr(result, 'race', {}),
                    "emotion": getattr(result, 'emotion', {}),
                    "accessories": getattr(result, 'accessories', {}),
                    "quality_metrics": getattr(result, 'quality_metrics', {}),
                    "quality_score": float(getattr(result, 'quality_score', 0.0)),
                    "landmarks": getattr(result, 'landmarks', None)
                }
                response.append(face_data)
                logger.debug(f"[API] Processed face {i}: {face_data}")
            except Exception as e:
                logger.error(f"[API] Error serializing face {i}: {str(e)}", exc_info=True)
        
        logger.info(f"[API] Successfully processed {len(response)} faces")
        
        # Save the final response for inspection
        import json
        response_path = os.path.join(debug_dir, 'api_response.json')
        with open(response_path, 'w') as f:
            json.dump({"results": response}, f, indent=2)
        logger.info(f"[API] Saved API response to: {os.path.abspath(response_path)}")
        
        return JSONResponse(content={"results": response})
        
    except Exception as e:
        logger.error(f"[API] Error in analysis: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Analysis failed", "details": str(e)}
        )

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
