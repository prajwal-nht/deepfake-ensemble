
"""
Video Processor Module
Handles video file input and frame extraction.
"""
import cv2
import numpy as np
from typing import Generator, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Process video files and extract frames for analysis."""
    
    def __init__(self, frame_interval: int = 10, target_size: Tuple[int, int] = (640, 480)):
        """
        Initialize video processor.
        
        Args:
            frame_interval: Extract one frame every N frames
            target_size: Target size for resizing frames (width, height)
        """
        self.frame_interval = frame_interval
        self.target_size = target_size
        
    def extract_frames(self, video_path: str) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to video file
            
        Yields:
            Tuple of (frame_number, frame_data)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return
            
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % self.frame_interval == 0:
                # Resize frame for consistent processing
                frame = cv2.resize(frame, self.target_size)
                yield frame_count, frame
                
            frame_count += 1
            
        cap.release()
        logger.info(f"Processed {frame_count} frames from {video_path}")

    @staticmethod
    def get_video_properties(video_path: str) -> Optional[dict]:
        """Get video properties like FPS, duration, etc."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'duration_seconds': duration,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
