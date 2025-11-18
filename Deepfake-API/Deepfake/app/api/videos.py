from typing import List, Dict, Any
from datetime import datetime
from ..schemas.video import VideoResponse
import os
import json

# In a real application, this would be a database
VIDEO_DB_PATH = "video_database.json"

def load_video_db() -> Dict[str, Any]:
    """Load video database from file"""
    if os.path.exists(VIDEO_DB_PATH):
        with open(VIDEO_DB_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_video_db(db: Dict[str, Any]) -> None:
    """Save video database to file"""
    with open(VIDEO_DB_PATH, 'w') as f:
        json.dump(db, f)

async def get_user_videos(user_id: str) -> List[VideoResponse]:
    """
    Get all videos related to a specific user.
    
    Args:
        user_id: ID of the user
        
    Returns:
        List of VideoResponse objects
    """
    try:
        db = load_video_db()
        user_videos = db.get(user_id, [])
        
        return [
            VideoResponse(
                video_id=video["video_id"],
                user_id=user_id,
                video_url=video["video_url"],
                is_deepfake=video["is_deepfake"],
                confidence=video["confidence"],
                processed_at=datetime.fromisoformat(video["processed_at"]),
                face_matches=video.get("face_matches"),
                frame_count=video["frame_count"],
                processed_frames=video["processed_frames"]
            )
            for video in user_videos
        ]
        
    except Exception as e:
        raise Exception(f"Error retrieving user videos: {str(e)}")

async def add_video_to_db(user_id: str, video_data: Dict[str, Any]) -> None:
    """
    Add a new video to the database.
    
    Args:
        user_id: ID of the user
        video_data: Video data to add
    """
    try:
        db = load_video_db()
        if user_id not in db:
            db[user_id] = []
        
        db[user_id].append(video_data)
        save_video_db(db)
        
    except Exception as e:
        raise Exception(f"Error adding video to database: {str(e)}") 