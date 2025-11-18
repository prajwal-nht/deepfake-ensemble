"""
FastAPI endpoint for parallel deepfake detection.
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

# Import our parallel processor
from models.parallel_processor import ParallelProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the router without prefix or tags - they'll be set in main.py
router = APIRouter()

# Initialize the processor (lazy load on first request)
_processor = None

def get_processor():
    """Get or initialize the parallel processor."""
    global _processor
    if _processor is None:
        logger.info("Initializing parallel processor...")
        _processor = ParallelProcessor()
    return _processor

@router.post("/detect/image", response_model=Dict[str, Any])
async def detect_image(file: UploadFile = File(...), confidence_threshold: float = 0.5):
    """
    Process an image through the parallel detection pipeline.
    
    Args:
        file: Uploaded image file
        confidence_threshold: Threshold for binary classification (default: 0.5)
        
    Returns:
        JSON with detection results from all models
    """
    try:
        # Validate confidence threshold
        if not 0 < confidence_threshold < 1:
            raise HTTPException(status_code=400, detail="Confidence threshold must be between 0 and 1")
            
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
            
        # Process the image
        start_time = time.time()
        results = await get_processor().process_image(image)
        processing_time = time.time() - start_time
        
        # Process results
        processed_results = {}
        successful_models = 0
        total_confidence = 0.0
        
        for model_name, result in results.items():
            if result.get('success') and 'probabilities' in result and result['probabilities']:
                try:
                    probs = result['probabilities']
                    
                    # Handle different probability formats
                    if isinstance(probs, (list, np.ndarray)) and len(probs) > 0:
                        # If it's a batch, take the first item
                        if isinstance(probs, np.ndarray) and len(probs.shape) > 1 and probs.shape[0] == 1:
                            probs = probs[0]
                        
                        # For binary classification, use the second probability (index 1) as the 'real' score
                        if len(probs) == 2:  # Binary classification
                            confidence = float(probs[1])  # Probability of being real
                        else:  # Multi-class, use max probability
                            confidence = float(np.max(probs))
                            
                        # Calculate model's decision
                        is_real = confidence > confidence_threshold
                        
                        processed_results[model_name] = {
                            'success': True,
                            'is_real': is_real,
                            'confidence': confidence,
                            'probabilities': probs.tolist() if hasattr(probs, 'tolist') else probs,
                            'model_decision': 'real' if is_real else 'fake'
                        }
                        successful_models += 1
                        total_confidence += confidence
                        continue
                        
                except Exception as e:
                    logger.error(f"Error processing results for {model_name}: {str(e)}", exc_info=True)
            
            # If we get here, there was an error or no probabilities
            processed_results[model_name] = {
                'success': result.get('success', False),
                'error': result.get('error', 'No probabilities available')
            }
        
        # Calculate overall metrics
        avg_confidence = total_confidence / successful_models if successful_models > 0 else 0.0
        
        # Determine overall decision (fake if average confidence < threshold)
        overall_decision = "real" if avg_confidence > confidence_threshold else "fake"
        
        # Count individual model decisions
        model_decisions = [r.get('model_decision') for r in processed_results.values() 
                         if r.get('success', False) and 'model_decision' in r]
        real_votes = model_decisions.count('real')
        fake_votes = model_decisions.count('fake')
        
        # Calculate model agreement (percentage of models that agree with the overall decision)
        agreeing_models = real_votes if overall_decision == "real" else fake_votes
        agreement_percentage = (agreeing_models / successful_models * 100) if successful_models > 0 else 0.0
        
        return {
            "status": "success",
            "metadata": {
                "filename": file.filename,
                "processing_time_seconds": round(processing_time, 4),
                "models_run": len(results),
                "successful_models": successful_models,
                "confidence_threshold": confidence_threshold
            },
            "results": {
                "overall_decision": overall_decision,
                "average_confidence": round(avg_confidence, 4),
                "model_agreement_percentage": round(agreement_percentage, 2),
                "votes": {
                    "real": real_votes,
                    "fake": fake_votes
                },
                "model_results": processed_results
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect/video", response_model=Dict[str, Any])
async def detect_video(
    file: UploadFile = File(...),
    frame_skip: int = 10,
    max_frames: int = 100,
    confidence_threshold: float = 0.5
):
    """
    Process a video file through the parallel detection pipeline.
    
    Args:
        file: Video file to process (MP4, AVI, MOV, MKV)
        frame_skip: Number of frames to skip between processing (default: 10)
        max_frames: Maximum number of frames to process (default: 100)
        confidence_threshold: Threshold for binary classification (default: 0.5)
        
    Returns:
        Dict containing detection results and metadata with the following structure:
        {
            "status": "success" | "error",
            "metadata": {
                "filename": str,
                "duration_seconds": float,
                "total_frames": int,
                "processed_frames": int,
                "frame_skip": int,
                "resolution": "WxH",
                "fps": float
            },
            "processing_stats": {
                "total_time_seconds": float,
                "processing_fps": float,
                "frames_processed_per_second": float
            },
            "results": {
                "overall_prediction": "real" | "fake",
                "overall_confidence": float,
                "model_agreement_percentage": float,
                "votes": {
                    "real": int,
                    "fake": int
                },
                "frame_level_results": List[Dict]  # First 10 frames
            },
            "error": str  # Only present if status is "error"
        }
    """
    start_time = time.time()
    temp_file = None
    cap = None
    
    try:
        logger.info(f"Starting video detection for file: {file.filename}")
        
        # Validate file type
        if not file.filename or not hasattr(file, 'filename') or not file.filename.strip():
            raise HTTPException(
                status_code=400,
                detail="No file provided or invalid filename"
            )
            
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: {file.filename}. Supported formats: .mp4, .avi, .mov, .mkv"
            )
        
        # Validate confidence threshold
        if not 0 < confidence_threshold < 1:
            raise HTTPException(
                status_code=400, 
                detail=f"Confidence threshold must be between 0 and 1, got {confidence_threshold}"
            )
        
        # Validate frame_skip and max_frames
        frame_skip = max(1, int(frame_skip))  # Ensure at least 1 and convert to int
        max_frames = max(1, int(max_frames))  # Ensure at least 1 and convert to int
        
        logger.info(f"Video processing parameters - frame_skip: {frame_skip}, max_frames: {max_frames}, confidence_threshold: {confidence_threshold}")
            
        logger.info(f"Starting video processing: {file.filename}")
        
        try:
            # Create temp directory if it doesn't exist
            temp_dir = "temp_uploads"
            try:
                os.makedirs(temp_dir, exist_ok=True)
                logger.info(f"Created/verified temp directory: {os.path.abspath(temp_dir)}")
                
                # Generate a safe filename
                safe_filename = "".join(c if c.isalnum() or c in '._-' else '_' for c in file.filename)
                temp_file_path = os.path.join(temp_dir, f"{int(time.time())}_{safe_filename}")
                
                logger.info(f"Saving uploaded file to: {temp_file_path}")
                
                try:
                    # Reset file pointer to start in case it was read before
                    file.file.seek(0)
                    
                    # Verify file has content
                    first_chunk = file.file.read(1024)
                    if not first_chunk:
                        raise ValueError("Uploaded file is empty")
                    
                    # Save file in chunks to handle large files
                    with open(temp_file_path, "wb") as temp_file:
                        # Write the first chunk we read
                        temp_file.write(first_chunk)
                        # Copy the rest of the file
                        shutil.copyfileobj(file.file, temp_file)
                        
                    file_size = os.path.getsize(temp_file_path)
                    logger.info(f"Successfully saved file. Size: {file_size} bytes")
                    
                    # Verify file is a valid video
                    cap_test = cv2.VideoCapture(temp_file_path)
                    if not cap_test.isOpened():
                        raise ValueError("Saved file is not a valid video or is corrupted")
                    
                    # Test reading first frame
                    ret, frame = cap_test.read()
                    cap_test.release()
                    
                    if not ret or frame is None:
                        raise ValueError("Could not read first frame from the saved video file")
                        
                except Exception as e:
                    # Clean up the temporary file if there was an error
                    if os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                        except Exception as cleanup_error:
                            logger.error(f"Error cleaning up temporary file: {str(cleanup_error)}")
                    raise e
                
            except Exception as e:
                logger.error(f"Error saving temporary file: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save uploaded file: {str(e)}"
                )
            
            # Open video file with better error handling and resource cleanup
            logger.info(f"Attempting to open video file: {temp_file_path}")
            cap = cv2.VideoCapture(temp_file_path)
            try:
                if not cap.isOpened():
                    error_msg = f"Failed to open video file: {temp_file_path}. "
                    error_msg += f"File exists: {os.path.exists(temp_file_path)}, "
                    error_msg += f"File size: {os.path.getsize(temp_file_path) if os.path.exists(temp_file_path) else 0} bytes"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                    
                # Get video properties with validation
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0
                
                logger.info(f"Video properties - FPS: {fps}, Width: {width}, Height: {height}, "
                          f"Total Frames: {total_frames}, Duration: {duration:.2f}s")
                
                if fps <= 0 or width <= 0 or height <= 0:
                    error_msg = f"Invalid video properties - FPS: {fps}, Width: {width}, Height: {height}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                if total_frames <= 0:
                    # Try alternative method to get frame count
                    try:
                        # Try to get frame count by seeking to end
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        total_frames = 0
                        while cap.grab():
                            total_frames += 1
                        if total_frames <= 0:
                            raise ValueError("Could not determine frame count")
                        # Reset to beginning
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    except Exception as e:
                        logger.warning(f"Could not determine exact frame count: {str(e)}")
                        # Fall back to a reasonable default
                        total_frames = max_frames * frame_skip
                
                logger.info(f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s")
                
                # Initialize processor
                processor = get_processor()
                
                # Initialize results
                frame_results = []
                model_votes = {"real": 0, "fake": 0}
                frame_processing_times = []
                all_model_confidences = []
                
                frame_count = 0
                processed_frames = 0
                
                # Log initial frame processing state
                logger.info(f"Starting frame processing with {total_frames} total frames")
                logger.info(f"Frame skip: {frame_skip}, Max frames: {max_frames}")
                logger.info(f"Video resolution: {width}x{height}, FPS: {fps}")
                
                logger.info(f"Starting frame processing (max {max_frames} frames, skipping {frame_skip} frames between processing)")
                logger.info(f"Video properties - FPS: {fps}, Total Frames: {total_frames}, Resolution: {width}x{height}")
                
                frame_count = 0
                while cap.isOpened() and processed_frames < max_frames:
                    logger.debug(f"Attempting to read frame {frame_count}")
                    
                    # Try reading frame with retry logic
                    retry_count = 0
                    max_retries = 3
                    frame = None
                    
                    while retry_count < max_retries:
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            break
                            
                        retry_count += 1
                        logger.warning(f"Failed to read frame {frame_count}, retry {retry_count}/{max_retries}")
                        # Try seeking to the next frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count + 1)
                    
                    if not ret or frame is None:
                        logger.info(f"Failed to read frame {frame_count} after {max_retries} attempts")
                        if frame_count == 0:
                            # If we can't read the first frame, the video might be corrupted
                            raise ValueError(f"Failed to read first frame from video. The video might be corrupted or in an unsupported format.")
                        logger.info(f"Reached end of video at frame {frame_count}")
                        break
                        
                    logger.debug(f"Successfully read frame {frame_count}, shape: {frame.shape if frame is not None else 'None'}")
                        
                    # Skip frames based on frame_skip
                    if frame_count % frame_skip != 0:
                        logger.debug(f"Skipping frame {frame_count} (frame_skip={frame_skip})")
                        frame_count += 1
                        continue
                        
                    logger.info(f"Processing frame {frame_count} (processed: {processed_frames + 1}/{max_frames})")
                    
                    frame_start_time = time.time()
                    
                    try:
                        logger.debug(f"Processing frame {frame_count}")
                        
                        try:
                            logger.info(f"Processing frame {frame_count} with shape: {frame.shape}")
                            result = await processor.process_image(frame)
                            frame_processing_time = time.time() - frame_start_time
                            frame_processing_times.append(frame_processing_time)
                            logger.info(f"Frame {frame_count} processed in {frame_processing_time:.4f}s")
                        except Exception as e:
                            logger.error(f"Error processing frame {frame_count}: {str(e)}", exc_info=True)
                            raise
                            
                        # Process model results for this frame
                        frame_result = {
                            'frame_number': frame_count,
                            'timestamp': frame_count / fps,
                            'models': {}
                        }
                        
                        # Process each model's output
                        model_results = result.get('results', {})
                        logger.info(f"Processing model outputs for frame {frame_count}. Raw results: {model_results}")
                        
                        if not model_results:
                            logger.warning(f"No model results found in frame {frame_count} processing output")
                            # Try to get results from the root of the result if not in 'results' key
                            model_results = {k: v for k, v in result.items() if k != 'results' and k in processor.available_models}
                            logger.info(f"Found {len(model_results)} models in root of result")
                        
                        for model_id, model_output in model_results.items():
                            try:
                                if not isinstance(model_output, dict):
                                    logger.warning(f"Model {model_id} output is not a dictionary: {model_output}")
                                    model_output = {
                                        'success': True,
                                        'prediction': str(model_output),
                                        'confidence': 0.0
                                    }
                                
                                if not model_output.get('success', True):
                                    error_msg = model_output.get('error', 'Unknown error')
                                    logger.error(f"Model {model_id} failed for frame {frame_count}: {error_msg}")
                                    frame_result['models'][model_id] = {
                                        'success': False,
                                        'error': error_msg
                                    }
                                    continue
                                    
                                # Extract prediction and confidence
                                prediction = model_output.get('prediction')
                                confidence = float(model_output.get('confidence', 0.0))
                                
                                frame_result['models'][model_id] = {
                                    'success': True,
                                    'prediction': prediction,
                                    'confidence': confidence
                                }
                                
                            except Exception as e:
                                logger.error(f"Error processing model {model_id} output: {str(e)}", exc_info=True)
                                frame_result['models'][model_id] = {
                                    'success': False,
                                    'error': f"Error processing output: {str(e)}"
                                }
                            
                            try:
                                probs = model_output.get('probabilities', [])
                                
                                # Determine if real or fake based on probabilities
                                if probs and len(probs) >= 2:
                                    # Assuming binary classification: [real_prob, fake_prob]
                                    real_prob = float(probs[0]) if len(probs) > 0 else 0.0
                                    fake_prob = float(probs[1]) if len(probs) > 1 else 0.0
                                    
                                    is_real = real_prob > fake_prob
                                    confidence = max(real_prob, fake_prob)
                                    
                                    frame_result['models'][model_id] = {
                                        'success': True,
                                        'is_real': bool(is_real),
                                        'confidence': float(confidence),
                                        'probabilities': [float(p) for p in probs]
                                    }
                                    
                                    # Update overall statistics
                                    if confidence >= confidence_threshold:
                                        if is_real:
                                            model_votes['real'] += 1
                                        else:
                                            model_votes['fake'] += 1
                                    all_model_confidences.append(confidence)
                                    
                                else:
                                    frame_result['models'][model_id] = {
                                        'success': False,
                                        'error': 'Invalid probability output format'
                                    }
                                    
                            except Exception as model_error:
                                logger.error(f"Error processing model {model_id} output: {str(model_error)}")
                                frame_result['models'][model_id] = {
                                    'success': False,
                                    'error': f'Error processing model output: {str(model_error)}'
                                }
                        
                        # Add frame result if we have any successful model outputs
                        if any(m.get('success', False) for m in frame_result['models'].values()):
                            frame_results.append(frame_result)
                            processed_frames += 1
                        
                        frame_count += 1
                        
                    except Exception as frame_error:
                        logger.error(f"Error processing frame {frame_count}:", exc_info=True)
                        frame_count += 1
                        # If we haven't processed any frames yet, fail fast
                        if processed_frames == 0 and frame_count >= min(10, max_frames):
                            logger.error("Failed to process any frames successfully. Aborting.")
                            raise HTTPException(
                                status_code=400,
                                detail=f"Failed to process any frames. The video might be corrupted or in an unsupported format. Error: {str(frame_error)}"
                            )
                        continue
                
                # Calculate overall statistics
                total_votes = sum(model_votes.values())
                if total_votes > 0:
                    overall_prediction = 'real' if model_votes['real'] > model_votes['fake'] else 'fake'
                    agreement_percentage = (max(model_votes.values()) / total_votes) * 100
                    avg_confidence = sum(all_model_confidences) / len(all_model_confidences) if all_model_confidences else 0.0
                else:
                    overall_prediction = 'unknown'
                    agreement_percentage = 0.0
                    avg_confidence = 0.0
                
                # Prepare response
                response = {
                    'status': 'success',
                    'metadata': {
                        'filename': file.filename,
                        'duration_seconds': round(duration, 2),
                        'total_frames': total_frames,
                        'processed_frames': processed_frames,
                        'frame_skip': frame_skip,
                        'resolution': f"{width}x{height}",
                        'fps': round(fps, 2),
                        'confidence_threshold': confidence_threshold
                    },
                    'processing_stats': {
                        'total_time_seconds': round(time.time() - start_time, 2),
                        'avg_processing_time_per_frame': round(sum(frame_processing_times) / len(frame_processing_times), 4) if frame_processing_times else 0,
                        'frames_processed_per_second': len(frame_processing_times) / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                    },
                    'results': {
                        'overall_prediction': overall_prediction,
                        'overall_confidence': round(avg_confidence, 4),
                        'model_agreement_percentage': round(agreement_percentage, 2),
                        'votes': model_votes,
                        'frame_level_results': frame_results[:10]  # Return first 10 frames for inspection
                    }
                }
                    
                logger.info(f"Video processing completed in {time.time() - start_time:.2f} seconds")
                logger.info(f"Overall prediction: {overall_prediction} (confidence: {avg_confidence:.4f})")
                
                # Clean up resources
                if 'cap' in locals() and cap is not None:
                    cap.release()
                
                # Remove temporary file
                if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        logger.info(f"Cleaned up temporary file: {temp_file_path}")
                    except Exception as cleanup_error:
                        logger.error(f"Error cleaning up temporary file: {str(cleanup_error)}")
                
                # Ensure we have valid results
                if not frame_results:
                    raise HTTPException(
                        status_code=400,
                        detail="No frames were successfully processed. The video might be corrupted or in an unsupported format."
                    )
                
                # Calculate overall statistics using frame decisions
                total_real = sum(1 for f in frame_results if f.get('frame_decision') == 'real')
                total_fake = len(frame_results) - total_real
                
                if total_real + total_fake == 0:
                    overall_prediction = 'unknown'
                    agreement_percentage = 0.0
                    avg_confidence = 0.0
                else:
                    overall_prediction = 'real' if total_real > total_fake else 'fake'
                    agreement_percentage = (max(total_real, total_fake) / (total_real + total_fake)) * 100
                    avg_confidence = sum(f.get('confidence', 0) for f in frame_results) / len(frame_results)
                            # Calculate total processing time
                total_processing_time = time.time() - start_time
                
                # Calculate processing statistics
                try:
                    avg_frame_processing_time = total_processing_time / len(frame_results) if frame_results else 0
                    processing_fps = len(frame_results) / total_processing_time if total_processing_time > 0 else 0
                    
                    # Prepare final response
                    response = {
                        'status': 'success',
                        'metadata': {
                            'filename': file.filename,
                            'duration_seconds': round(duration, 2) if 'duration' in locals() else 0,
                            'total_frames': total_frames if 'total_frames' in locals() else 0,
                            'processed_frames': len(frame_results),
                            'frame_skip': frame_skip,
                            'resolution': f"{width}x{height}" if all(v in locals() for v in ['width', 'height']) else 'unknown',
                            'fps': round(fps, 2) if 'fps' in locals() else 0,
                            'confidence_threshold': confidence_threshold
                        },
                        'processing_stats': {
                            'total_time_seconds': round(total_processing_time, 2),
                            'avg_frame_processing_seconds': round(avg_frame_processing_time, 4),
                            'processing_fps': round(processing_fps, 2),
                            'frames_processed_per_second': round(processing_fps, 2)
                        },
                        'results': {
                            'overall_prediction': overall_prediction,
                            'overall_confidence': round(avg_confidence, 4),
                            'model_agreement_percentage': round(agreement_percentage, 2),
                            'votes': {
                                'real': total_real,
                                'fake': total_fake
                            },
                            'frame_level_results': frame_results[:10]  # Only return first 10 frames
                        }
                    }
                    
                    # Add note if frame results were truncated
                    if len(frame_results) > 10:
                        response['results']['note'] = (
                            f"Frame-level results truncated to 10 frames. "
                            f"Processed {len(frame_results)} frames total."
                        )
                    
                    logger.info(f"Video processing completed in {total_processing_time:.2f} seconds")
                    logger.info(f"Overall prediction: {overall_prediction} (confidence: {avg_confidence:.4f})")
                    
                    return response
                    
                except Exception as e:
                    logger.error(f"Error preparing response: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error preparing response: {str(e)}"
                    )
                
            except HTTPException as he:
                # Re-raise HTTP exceptions as-is
                logger.error(f"HTTP error in video processing: {str(he.detail)}")
                raise
                
            except Exception as e:
                # Log the full exception with traceback
                logger.error(f"Unexpected error processing video: {str(e)}", exc_info=True)
                
                # Clean up any open resources
                if 'cap' in locals() and cap is not None and cap.isOpened():
                    cap.release()
                if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        logger.info(f"Cleaned up temporary file: {temp_file_path}")
                    except Exception as cleanup_error:
                        logger.error(f"Error cleaning up temporary file: {str(cleanup_error)}")
                
                # Provide a more detailed error message
                error_detail = f"An error occurred during video processing: {str(e)}"
                if "No such file or directory" in str(e):
                    error_detail = "Temporary file could not be created. Check disk space and permissions."
                elif "Invalid data" in str(e) or "could not find codec parameters" in str(e).lower():
                    error_detail = "The video file appears to be corrupted, in an unsupported format, or uses an unsupported codec."
                elif "OpenCV" in str(e):
                    error_detail = "Video processing error. The file might be corrupted or use an unsupported codec."
                
                logger.error(f"Video processing failed: {error_detail}")
                
                raise HTTPException(
                    status_code=500,
                    detail=error_detail
                )
            
            # Add note if frame results were truncated
            if len(frame_results) > 10:
                response["results"]["note"] = (
                    f"Frame-level results truncated to 10 frames. "
                    f"Processed {len(frame_results)} frames total."
                )
            
            logger.info(f"Video processing completed in {total_processing_time:.2f} seconds")
            logger.info(f"Overall prediction: {overall_prediction} (confidence: {avg_confidence:.4f}")
            
            return response
            
        except HTTPException as he:
            # Re-raise HTTP exceptions as-is
            logger.error(f"HTTP error in video processing: {str(he.detail)}")
            raise
            
        except Exception as e:
            # Log the full exception with traceback
            logger.error(f"Unexpected error processing video: {str(e)}", exc_info=True)
            
            # Clean up any open resources
            if 'cap' in locals() and cap is not None and cap.isOpened():
                cap.release()
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up temporary file: {str(cleanup_error)}")
            
            # Provide a more detailed error message
            error_detail = f"An error occurred during video processing: {str(e)}"
            if "No such file or directory" in str(e):
                error_detail = "Temporary file could not be created. Check disk space and permissions."
            elif "Invalid data" in str(e) or "could not find codec parameters" in str(e).lower():
                error_detail = "The video file appears to be corrupted, in an unsupported format, or uses an unsupported codec."
            elif "OpenCV" in str(e):
                error_detail = "Video processing error. The file might be corrupted or use an unsupported codec."
            
            logger.error(f"Video processing failed: {error_detail}")
            
            raise HTTPException(
                status_code=500,
                detail=error_detail
            )
            
        finally:
            # Clean up resources
            if cap is not None:
                cap.release()
            if os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.error(f"Error deleting temporary file: {str(e)}")
                    
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error in video processing: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred: {str(e)}"
        )

@router.get("/models", response_model=Dict[str, Any])
async def list_models():
    """List all available models and their status."""
    try:
        processor = get_processor()
        
        # Enhanced debugging
        logger.info(f"Available models type: {type(processor.available_models)}")
        logger.info(f"Available models content: {processor.available_models}")
        logger.info(f"Processor type: {type(processor)}")
        logger.info(f"Processor dir: {dir(processor)}")
        
        # Try to get available_models in a safer way
        try:
            if hasattr(processor, 'available_models'):
                available_models = processor.available_models
                logger.info(f"Got available_models: {available_models}")
                
                # If it's a string, try to evaluate it as a list
                if isinstance(available_models, str):
                    try:
                        import ast
                        available_models = ast.literal_eval(available_models)
                        logger.info(f"Parsed available_models from string: {available_models}")
                    except (ValueError, SyntaxError) as e:
                        logger.warning(f"Could not parse available_models string: {e}")
                        available_models = [available_models]
                
                # Convert to a dictionary format
                if isinstance(available_models, (list, tuple)):
                    models_info = {
                        f"model_{i}": {
                            "name": str(model),
                            "status": "loaded"
                        }
                        for i, model in enumerate(available_models)
                    }
                elif isinstance(available_models, dict):
                    models_info = {
                        str(model_id): {
                            "name": str(model_info) if not isinstance(model_info, dict) 
                                      else model_info.get("name", str(model_info)),
                            "status": "loaded"
                        }
                        for model_id, model_info in available_models.items()
                    }
                else:
                    models_info = {
                        "single_model": {
                            "name": str(available_models),
                            "status": "loaded"
                        }
                    }
                
                return {
                    "status": "success",
                    "models": models_info,
                    "device": str(processor.device) if hasattr(processor, 'device') else "unknown"
                }
            else:
                return {
                    "status": "error",
                    "message": "No available_models attribute found on processor",
                    "available_attributes": dir(processor)
                }
                
        except Exception as inner_e:
            logger.error(f"Error processing available_models: {str(inner_e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error processing available_models: {str(inner_e)}",
                "available_models_type": str(type(processor.available_models))
            }
            
    except Exception as e:
        logger.error(f"Error in list_models: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
            "error_type": type(e).__name__
        }

# Export the router
__all__ = ["router"]

# This ensures the router is properly imported when the module is loaded
print("Parallel detection router initialized")
