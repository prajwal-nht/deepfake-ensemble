from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class VideoResponse(BaseModel):
    video_id: str
    user_id: str
    video_url: str
    is_deepfake: bool
    confidence: float
    processed_at: datetime
    face_matches: Optional[list] = None
    frame_count: int
    processed_frames: int 