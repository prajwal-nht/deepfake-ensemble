import numpy as np
import torch
import asyncio
from typing import Dict, Any, List
import os
import cv2
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image

# Import model loading and inference functions
from models.pruthvi_ml import load_model as load_model1, run_inference as inference1
from models.model2 import load_model as load_model2, run_inference as inference2
from models.cross_efficient_vit import load_model as load_model3, run_inference as inference3
from models.efficient_net_auto_att import load_model as load_model4, run_inference as inference4
from models.xceptionnet import load_model as load_model5, run_inference as inference5

# Import preprocessing functions
from preprocess.preprocess_pruthvi_ml import preprocess_images as preprocess1
from preprocess.preprocess2 import preprocess_images as preprocess2
from preprocess.preprocess_cross_efficient_vit import preprocess_images as preprocess3
from preprocess.preprocess_efficientnet_xceptionnet import preprocess_images as preprocess4

# Model weights
WEIGHTS = [0.23, 0.23, 0.23, 0.23, 0.08]
THRESHOLD = 0.5

# Model paths
MODEL_PATHS = {
    "model1": "pretrained_models/model1.safetensors",
    "model2": "pretrained_models/model2.safetensors", 
    "model3": "pretrained_models/cross_efficient_vit.pth",
    "model4": "pretrained_models/EfficientNetAutoAttB4ST_FFPP.pth",
    "model5": "pretrained_models/Xception_FFPP.pth"
}

async def run_model(load_fn, inference_fn, preprocess_fn, image_list, model_path, device):
    model = load_fn(device, model_path)
    predictions = []
    loop = asyncio.get_running_loop()
    for image_path in image_list:
        input_tensor = preprocess_fn([image_path])
        pred = await loop.run_in_executor(None, inference_fn, model, input_tensor, device)
        if isinstance(pred, np.ndarray):
            predictions.append(pred[0])
        elif isinstance(pred, (list, tuple)):
            predictions.append(pred[0])
        else:
            predictions.append(pred)
    return np.array(predictions)

def compute_weighted_scores(preds_list, weights):
    preds_tf = tf.convert_to_tensor(preds_list, dtype=tf.float32)
    weights_tf = tf.convert_to_tensor(weights, dtype=tf.float32)
    weighted_scores_tf = tf.reduce_sum(preds_tf * weights_tf[:, None], axis=0)
    return weighted_scores_tf.numpy()

async def run_deepfake_detection(face_frames: List[str], output_dir: str = "temp_frames") -> Dict[str, Any]:
    """
    Run ensemble deepfake detection on face frames.
    
    Args:
        face_frames: List of base64 encoded face frames to analyze
        output_dir: Directory to save temporary frame images
        
    Returns:
        Dictionary containing detection results including:
        - confidence: Overall confidence score
        - predictions: List of predictions per frame
        - verdicts: List of verdicts per frame
    """
    # Create temp directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save frames as images
    image_list = []
    for i, frame_base64 in enumerate(face_frames):
        try:
            # Convert base64 to numpy array
            img_data = base64.b64decode(frame_base64)
            img = Image.open(BytesIO(img_data))
            frame = np.array(img)
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            frame_path = os.path.join(output_dir, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame_bgr)
            image_list.append(frame_path)
        except Exception as e:
            print(f"Error processing frame {i}: {str(e)}")
            continue

    if not image_list:
        # Clean up empty directory
        try:
            os.rmdir(output_dir)
        except Exception as e:
            print(f"Error removing empty temp directory: {str(e)}")
        return {"error": "No frames to analyze"}

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tasks = [
            run_model(load_model1, inference1, preprocess1, image_list, MODEL_PATHS["model1"], device),
            run_model(load_model2, inference2, preprocess2, image_list, MODEL_PATHS["model2"], device),
            run_model(load_model3, inference3, preprocess3, image_list, MODEL_PATHS["model3"], device),
            run_model(load_model4, inference4, preprocess4, image_list, MODEL_PATHS["model4"], device),
            run_model(load_model5, inference5, preprocess4, image_list, MODEL_PATHS["model5"], device)
        ]

        preds_list = await asyncio.gather(*tasks)
        final_scores = compute_weighted_scores(preds_list, WEIGHTS)
        confidence = round(np.average(final_scores) * 100, 2)

        results = []
        for idx, score in enumerate(final_scores):
            verdict = "Deepfake" if score > THRESHOLD else "Real"
            results.append({
                "frame_index": idx,
                "score": float(score),
                "verdict": verdict
            })

        return {
            "confidence": confidence,
            "results": results,
            "overall_verdict": "Deepfake" if confidence > (THRESHOLD * 100) else "Real"
        }
    except Exception as e:
        print(f"Error during model inference: {str(e)}")
        return {"error": f"Model inference failed: {str(e)}"}
    finally:
        # Cleanup temp files
        for img_path in image_list:
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
            except Exception as e:
                print(f"Error removing temp file {img_path}: {str(e)}")
        
        # Cleanup temp directory
        try:
            if os.path.exists(output_dir):
                # Remove any remaining files
                for file in os.listdir(output_dir):
                    try:
                        os.remove(os.path.join(output_dir, file))
                    except Exception as e:
                        print(f"Error removing file {file}: {str(e)}")
                # Remove the directory
                os.rmdir(output_dir)
        except Exception as e:
            print(f"Error cleaning up temp directory: {str(e)}")