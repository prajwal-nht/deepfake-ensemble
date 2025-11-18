import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from pinecone import Pinecone
import uuid
from dotenv import load_dotenv
from typing import Dict, Any
load_dotenv() 

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create or get the face embedding index
usrfc_index_name = "user-face-embed"
dimension = 512  # InsightFace embedding dimension

# Check if index exists, if not create it
if usrfc_index_name not in pc.list_indexes().names():
    pc.create_index(
        name=usrfc_index_name,
        dimension=dimension,
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )

usrfc_index = pc.Index(usrfc_index_name)

# Set environment variable to prevent downloads
os.environ['INSIGHTFACE_HOME'] = '/root/.insightface'
os.environ['INSIGHTFACE_DOWNLOAD_ROOT'] = 'https://nonexistent-domain.com'  # Will fail if tries to download

# Initialize FaceAnalysis with explicit model path
model_path = '/root/.insightface/models/buffalo_l'
print(f"Looking for models in: {model_path}")
print(f"Directory exists: {os.path.exists(model_path)}")
if os.path.exists(model_path):
    print(f"Model files: {os.listdir(model_path)}")

# Initialize with explicit path and settings
app = FaceAnalysis(
    name='buffalo_l',
    root=os.path.dirname(os.path.dirname(model_path)),  # Point to parent of 'models' directory
    allowed_modules=['detection', 'recognition'],
    providers=['CPUExecutionProvider']  # Use CPU only for now to simplify
)

# Prepare the model with specific settings
try:
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Successfully loaded FaceAnalysis model")
except Exception as e:
    print(f"Error preparing FaceAnalysis model: {e}")
    print(f"Models root: {app.models_root}")
    if hasattr(app, 'models'):
        print(f"Available models: {list(app.models.keys())}")
    raise

async def index_face(video_path: str) -> Dict[str, Any]:
    """
    Process a video file to extract and index face embeddings.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary containing status and face ID
    """
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "error", "message": "Could not open video file"}
        
        all_embeddings = []
        frame_count = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process every 10th frame
            if frame_count % 10 == 0:
                faces = app.get(frame)
                if faces:
                    for face in faces:
                        embedding = face['embedding']
                        all_embeddings.append(embedding)
                    processed_frames += 1
        
        cap.release()
        
        if not all_embeddings:
            return {"status": "error", "message": "No faces detected in video"}
            
        # Average all embeddings to get a single video embedding
        video_embedding = np.mean(np.array(all_embeddings), axis=0).tolist()
        
        # Generate a unique face ID
        face_id = str(uuid.uuid4())
        
        # Store in Pinecone
        usrfc_index.upsert(vectors=[(face_id, video_embedding)])
        
        return {
            "status": "success",
            "face_id": face_id,
            "frame_count": frame_count,
            "processed_frames": processed_frames
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)} 