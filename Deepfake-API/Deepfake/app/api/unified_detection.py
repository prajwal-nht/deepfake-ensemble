"""
Unified Deepfake Detection API Endpoint
"""
import os
import logging
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
import cv2
import numpy as np
from models.parallel_processor import ParallelProcessor
from models.ensemble import WeightedEnsemble
from pipeline.demographic_analyzer import DemographicAnalyzer

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize components
_processor = None
_ensemble = None
_demographic_analyzer = None

def get_processor():
    global _processor
    if _processor is None:
        _processor = ParallelProcessor()
    return _processor

def get_ensemble():
    """Get or initialize the weighted ensemble."""
    global _ensemble
    if _ensemble is None:
        logger.info("Initializing weighted ensemble...")
        processor = get_processor()
        # Access models through model_loader
        models = {model_id: processor.model_loader.models[model_id] for model_id in processor.model_loader.models}
        _ensemble = WeightedEnsemble(models)
    return _ensemble

def get_demographic_analyzer():
    global _demographic_analyzer
    if _demographic_analyzer is None:
        _demographic_analyzer = DemographicAnalyzer()
    return _demographic_analyzer

@router.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """Process image through complete pipeline."""
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get components
        processor = get_processor()
        ensemble = get_ensemble()
        demo_analyzer = get_demographic_analyzer()
        
        # Process the image using the processor's process_frame method
        result = processor.process_frame(image_rgb)
        
        # Format the response
        response = {'status': 'success', 'faces': []}
        
        # Handle face analyses if any faces were detected
        if 'face_analyses' in result and result['face_analyses']:
            for face in result['face_analyses']:
                if 'success' in face and not face['success']:
                    continue  # Skip failed face analyses
                    
                # Extract face information
                face_info = {
                    'bbox': face.get('bbox', []),
                    'landmarks': face.get('landmarks', []),
                    'prediction': face.get('prediction', 'unknown'),
                    'confidence': float(face.get('confidence', 0.0)),
                    'demographics': face.get('demographics', {})
                }
                response['faces'].append(face_info)
        
        # If no faces were detected but we have frame analysis, include it
        if not response['faces'] and 'frame_analysis' in result and result['frame_analysis'].get('success', False):
            response['frame_analysis'] = {
                'prediction': result['frame_analysis'].get('prediction', 'unknown'),
                'confidence': float(result['frame_analysis'].get('confidence', 0.0)),
                'note': result['frame_analysis'].get('note', 'Full frame analysis')
            }
            
        return response
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    frame_skip: int = 10,
    max_frames: int = 100
):
    """Process video through complete pipeline."""
    temp_video_path = None
    cap = None
    try:
        # Save the uploaded file to a temporary location
        temp_dir = tempfile.mkdtemp()
        temp_video_path = os.path.join(temp_dir, 'video.mp4')
        
        # Write the uploaded file content to the temporary file
        with open(temp_video_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Open the video file
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Initialize processor
        processor = get_processor()
        
        # Process frames
        frame_count = 0
        processed_frames = 0
        results = {
            'status': 'success',
            'video_metadata': {
                'fps': float(fps),
                'total_frames': total_frames,
                'duration_seconds': float(duration)
            },
            'frames_processed': 0,
            'faces_detected': 0,
            'predictions': []
        }
        
        while processed_frames < max_frames and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames according to frame_skip
            if frame_count % (frame_skip + 1) != 0:
                frame_count += 1
                continue
                
            # Process the frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = processor.process_frame(frame_rgb)
            
            # Format the frame result
            frame_result = {
                'frame_number': frame_count,
                'timestamp': frame_count / fps if fps > 0 else 0,
                'faces': []
            }
            
            # Process face analyses if any faces were detected
            if 'face_analyses' in result and result['face_analyses']:
                for face in result['face_analyses']:
                    if 'success' in face and not face['success']:
                        continue  # Skip failed face analyses
                        
                    face_info = {
                        'bbox': face.get('bbox', []),
                        'landmarks': face.get('landmarks', []),
                        'prediction': face.get('prediction', 'unknown'),
                        'confidence': float(face.get('confidence', 0.0)),
                        'demographics': face.get('demographics', {})
                    }
                    frame_result['faces'].append(face_info)
                    results['faces_detected'] += 1
            
            # If no faces but we have frame analysis, include it
            if not frame_result['faces'] and 'frame_analysis' in result and result['frame_analysis'].get('success', False):
                frame_result['frame_analysis'] = {
                    'prediction': result['frame_analysis'].get('prediction', 'unknown'),
                    'confidence': float(result['frame_analysis'].get('confidence', 0.0)),
                    'note': result['frame_analysis'].get('note', 'Full frame analysis')
                }
            
            results['predictions'].append(frame_result)
            results['frames_processed'] += 1
            processed_frames += 1
            frame_count += 1
        
        # Add summary statistics
        if results['predictions']:
            predictions = [p for frame in results['predictions'] 
                         for p in frame.get('faces', [])]
            if predictions:
                fake_count = sum(1 for p in predictions 
                               if p.get('prediction', '').lower() == 'fake')
                results['summary'] = {
                    'total_faces': len(predictions),
                    'fake_faces': fake_count,
                    'real_faces': len(predictions) - fake_count,
                    'fake_ratio': fake_count / len(predictions) if predictions else 0.0
                }
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
        
    finally:
        # Release resources
        if cap is not None:
            cap.release()
        
        # Clean up temporary files
        try:
            if temp_video_path and os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {str(e)}")
