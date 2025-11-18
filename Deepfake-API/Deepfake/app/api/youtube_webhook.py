import os
import cv2
import numpy as np
from pytubefix import YouTube
from pytubefix.cli import on_progress
from insightface.app import FaceAnalysis
from pinecone import Pinecone
import uuid
from datetime import datetime
from typing import Dict, Any, List, Tuple
import asyncio
# Assuming this is in the same directory or a resolvable path
from .deepfake_detection import run_deepfake_detection
import urllib.parse
import base64
from io import BytesIO
from PIL import Image

# Initialize Pinecone
# Ensure PINECONE_API_KEY environment variable is set
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create or get the face embedding index
usrfc_index_name = "user-face-embed"
dimension = 512  # InsightFace embedding dimension

# Check if index exists, if not create it
if usrfc_index_name not in pc.list_indexes().names():
    print(f"Index '{usrfc_index_name}' not found. Creating new index.")
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
else:
    print(f"Found existing index: '{usrfc_index_name}'")

usrfc_index = pc.Index(usrfc_index_name)

# Initialize FaceAnalysis model
try:
    # Use CPUExecutionProvider as a fallback if CUDA is not available
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("FaceAnalysis model initialized successfully.")
except Exception as e:
    print(f"Error initializing FaceAnalysis model: {e}")
    app = None


async def download_youtube_video(url: str) -> str:
    """Download YouTube video and return local path"""
    try:
        # Decode URL to handle potential encoding issues
        decoded_url = urllib.parse.unquote(url)
        print(f"Attempting to download from URL: {decoded_url}")

        # Create a temporary directory for uploads if it doesn't exist
        os.makedirs("temp_uploads", exist_ok=True)

        # Initialize YouTube object with progress callback
        yt = YouTube(decoded_url, on_progress_callback=on_progress)
        print(f"Video title: {yt.title}")
        print(f"Video length: {yt.length} seconds")

        # Get the highest resolution stream
        ys = yt.streams.get_highest_resolution()
        if not ys:
            raise Exception("No suitable video stream found.")

        # Generate a unique filename
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}.mp4"
        video_path = os.path.join("temp_uploads", filename)

        print(f"Downloading video to: {video_path}")
        # Download the video
        ys.download(output_path="temp_uploads", filename=filename)
        print(f"Video downloaded successfully: {video_path}")
        return video_path

    except Exception as e:
        # Catch any exception during download and provide informative error
        print(f"Error downloading video from {url}: {str(e)}")
        raise Exception(f"Failed to download YouTube video: {str(e)}")


async def extract_face_frames(video_path: str) -> List[Dict[str, Any]]:
    """
    Extract frames containing faces from video.
    Returns a list of dictionaries, each containing the face embedding and
    the base64 encoded string of the cropped face image.
    """
    if not app:
        print("FaceAnalysis model is not available. Cannot extract face frames.")
        return []

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    extracted_faces_data = []
    frame_number = 0
    processed_face_count = 0

    print(f"Starting face extraction from video: {video_path}")

    # Process every Nth frame to reduce processing time
    FRAME_PROCESSING_INTERVAL = 30

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            # Break the loop if no more frames are available
            break

        frame_number += 1
        # Skip frames based on the interval
        if frame_number % FRAME_PROCESSING_INTERVAL != 0:
            continue

        # print(f"Processing video frame number: {frame_number}") # Can be verbose

        try:
            # Check if the frame is valid
            if frame is None or frame.size == 0:
                print(f"Invalid frame {frame_number}. Skipping.")
                continue

            # Convert frame to RGB (InsightFace expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Detect faces in the frame
            detected_faces = app.get(frame_rgb)
            # print(f"Found {len(detected_faces)} faces in video frame {frame_number}") # Can be verbose

            if detected_faces:
                for face_obj in detected_faces:
                    try:
                        # Get the face embedding
                        embedding_vector = face_obj.embedding
                        if embedding_vector is None:
                            # print(f"Could not get embedding for a face in frame {frame_number}. Skipping face.")
                            continue

                        # Get face bounding box and add padding
                        x1, y1, x2, y2 = map(int, face_obj['bbox'])
                        padding = 20
                        img_h, img_w, _ = frame_rgb.shape
                        x1_pad = max(0, x1 - padding)
                        y1_pad = max(0, y1 - padding)
                        x2_pad = min(img_w, x2 + padding)
                        y2_pad = min(img_h, y2 + padding)

                        # Validate padded coordinates
                        if x2_pad <= x1_pad or y2_pad <= y1_pad:
                            # print(f"Invalid face coordinates after padding in frame {frame_number}. Skipping face.")
                            continue

                        # Crop the face region of interest (ROI)
                        face_roi = frame_rgb[y1_pad:y2_pad, x1_pad:x2_pad]

                        # Check if the cropped ROI is valid
                        if face_roi is None or face_roi.size == 0:
                            # print(f"Invalid face ROI in frame {frame_number}. Skipping face.")
                            continue

                        # Convert cropped face to base64 string
                        pil_image = Image.fromarray(face_roi)
                        buffered = BytesIO()
                        pil_image.save(buffered, format="JPEG", quality=95)
                        img_str_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                        # Store the extracted face data
                        extracted_faces_data.append({
                            'embedding': embedding_vector,
                            'crop_base64': img_str_base64,
                            'source_frame': frame_number # Optional: for tracing back
                        })
                        processed_face_count += 1

                    except Exception as e:
                        # Handle errors during processing of a single detected face
                        print(f"Error processing a detected face in frame {frame_number}: {str(e)}")
                        continue

        except Exception as e:
            # Handle errors during processing of a video frame
            print(f"Error processing video frame {frame_number} for face detection: {str(e)}")
            continue

    # Release the video capture object
    cap.release()
    effective_frames_processed = (frame_number // FRAME_PROCESSING_INTERVAL) if FRAME_PROCESSING_INTERVAL > 0 else frame_number
    print(f"Finished face extraction. Total video frames processed for detection: {effective_frames_processed}. Total individual face data extracted: {processed_face_count}")
    return extracted_faces_data


async def get_faces_for_deepfake_check(face_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    De-duplicate embeddings (intra-video), then query Pinecone.
    Returns a list of unique faces (embedding + crop_base64) that have
    a Pinecone match score >= DEEPFAKE_PINE_MATCH_THRESHOLD.
    """
    if not face_data_list:
        return []

    # Intra-video uniqueness: Identify representative faces (embedding + crop_base64)
    # This step ensures we don't query Pinecone multiple times for the same person within the video
    representative_faces_data: List[Dict[str, Any]] = []
    INTRA_THRESH = 0.75  # Threshold for considering faces as the same person within the video

    for current_face_data in face_data_list:
        current_emb = current_face_data['embedding']
        is_unique = True

        if not representative_faces_data:
            # If no representative faces yet, add the first one
            representative_faces_data.append(current_face_data)
        else:
            # Calculate similarity with existing representative faces
            similarities = [float(np.dot(current_emb, rep_face['embedding'])) for rep_face in representative_faces_data]
            # If the current face is similar to any representative face above the threshold, it's not unique
            if similarities and max(similarities) >= INTRA_THRESH:
                is_unique = False
            # If the face is unique, add it to the list of representative faces
            if is_unique:
                representative_faces_data.append(current_face_data)

    print(f"Identified {len(representative_faces_data)} unique representative faces in the video for Pinecone matching.")

    # Stricter threshold for sending to deepfake detection based on Pinecone match
    DEEPFAKE_PINE_MATCH_THRESHOLD = 0.6

    # List to store face data that meets the deepfake criteria (unique in video AND Pinecone match >= 0.6)
    faces_meeting_deepfake_criteria_data: List[Dict[str, Any]] = []

    # Query Pinecone with each representative face embedding
    for rep_face_data in representative_faces_data:
        emb_to_query = rep_face_data['embedding']
        try:
            query_response = usrfc_index.query(
                vector=emb_to_query.tolist(),
                top_k=1, # Get the single best match
                include_metadata=True
            )
            # Check if a match was found and if its score meets the deepfake threshold
            if query_response.matches:
                match = query_response.matches[0]

                if match.score is not None and match.score >= DEEPFAKE_PINE_MATCH_THRESHOLD:
                    # If the match score is high enough, add the original representative face data
                    # (which includes the crop_base64) to the list for deepfake detection.
                    faces_meeting_deepfake_criteria_data.append(rep_face_data)
                    print(f"Face matched Pinecone with score {match.score:.4f} >= {DEEPFAKE_PINE_MATCH_THRESHOLD}. Added for deepfake check.")
                # else:
                    # print(f"Face matched Pinecone with score {match.score:.4f} < {DEEPFAKE_PINE_MATCH_THRESHOLD}. Skipping deepfake check.")

        except Exception as e:
            # Handle errors during Pinecone query
            print(f"Error querying Pinecone with an embedding: {str(e)}")
            continue

    print(f"Identified {len(faces_meeting_deepfake_criteria_data)} unique faces meeting the deepfake criteria (Pinecone match >= {DEEPFAKE_PINE_MATCH_THRESHOLD}).")

    # Return only the list of faces that should be sent to deepfake detection
    return faces_meeting_deepfake_criteria_data


async def process_youtube_video(video_url: str) -> Dict[str, Any]:
    """
    Process a YouTube video for face matching and deepfake detection.
    Only unique faces with a high-confidence Pinecone match (>= 0.6) are sent to deepfake detection.
    """
    if not app:
        return {
            "status": "error",
            "message": "FaceAnalysis model not initialized. Cannot process video."
        }

    video_path = None
    try:
        # Download the video
        video_path = await download_youtube_video(video_url)
        # Extract all faces from the video frames
        all_extracted_faces_data = await extract_face_frames(video_path)

        # If no faces were extracted, return early
        if not all_extracted_faces_data:
            if video_path and os.path.exists(video_path):
                os.remove(video_path) # Clean up the downloaded video file
            return {
                "status": "success",
                "message": "No faces detected in video or failed to extract face data.",
                "video_id": str(uuid.uuid4()),
                "Deepfake_Result": {},
                "extracted_face_data_count": 0,
                "faces_sent_to_deepfake_count": 0,
                "processed_at": datetime.now().isoformat()
            }

        # Get the list of unique faces that meet the 0.6 Pinecone match threshold
        faces_for_deepfake_input_data = await get_faces_for_deepfake_check(all_extracted_faces_data)

        deepfake_results = {}
        # Prepare face crops for deepfake detection using ONLY faces that met the 0.6 Pinecone match criteria
        face_crops_for_deepfake = [
            face_data['crop_base64']
            for face_data in faces_for_deepfake_input_data
            if 'crop_base64' in face_data
        ]

        # Run deepfake detection if there are faces meeting the criteria
        if face_crops_for_deepfake:
            print(f"Proceeding to deepfake detection with {len(face_crops_for_deepfake)} face crops that met Pinecone match threshold >= 0.6.")
            # Call the deepfake detection function
            deepfake_results = await run_deepfake_detection(face_crops_for_deepfake)

            if "error" in deepfake_results:
                print(f"Error during deepfake detection: {deepfake_results['error']}")
        else:
            print("No face crops met the criteria for deepfake detection (unique in video AND Pinecone match >= 0.6).")


        # Clean up the downloaded video file
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as cleanup_e:
                print(f"Error during cleanup of video_path: {cleanup_e}")

        # Return the processing results
        return {
            "status": "success",
            "video_id": str(uuid.uuid4()), # Generate a unique ID for this processing instance
            "Deepfake_Result": deepfake_results, # Results from deepfake detection
            # Removed the general face_matches list based on 0.4 threshold as requested
            "extracted_face_data_count": len(all_extracted_faces_data), # Total faces extracted before uniqueness/Pinecone filtering
            "faces_sent_to_deepfake_count": len(face_crops_for_deepfake), # Count of faces actually sent to deepfake detection
            "processed_at": datetime.now().isoformat() # Timestamp of processing
        }

    except Exception as e:
        # Catch any unexpected errors during the process
        print(f"An error occurred in process_youtube_video: {str(e)}")
        # Attempt to clean up the video file even if an error occurred
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as cleanup_e:
                print(f"Error during cleanup of video_path after error: {cleanup_e}")
        # Return an error status
        return {
            "status": "error",
            "message": str(e),
            "video_id": str(uuid.uuid4()) # Still provide a unique ID for the failed attempt
        }

