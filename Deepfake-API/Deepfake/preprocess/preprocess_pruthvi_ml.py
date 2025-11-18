import cv2
import numpy as np
import json
import torch

PREPROCESS_CONFIG_FILE_PATH = "preprocess/preprocessor_config_pruthvi_ml.json"
def load_preprocessor_config(config_path=PREPROCESS_CONFIG_FILE_PATH):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def preprocess_images(image_paths, config_path=PREPROCESS_CONFIG_FILE_PATH):
    """
    Loads images from the given paths or processes NumPy arrays and applies the configured preprocessing steps.
    Returns a tensor of shape (N, C, H, W).
    """
    config = load_preprocessor_config(config_path)
    processed_images = []
    for path_or_array in image_paths:
        # Handle both file paths and NumPy arrays
        if isinstance(path_or_array, str):
            img = cv2.imread(path_or_array)
            if img is None:
                raise ValueError(f"Image not loaded: {path_or_array}")
        else:
            img = path_or_array.copy()
            
        if config.get("do_convert_rgb", True):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if config.get("do_resize", True):
            size = config["size"]
            img = cv2.resize(img, (size["width"], size["height"]))
        if config.get("do_normalize", True):
            img = img.astype("float32")
            if config.get("do_rescale", True):
                factor = config.get("rescale_factor", 1.0/255)
                img = img * factor
        processed_images.append(img)
    
    images_np = np.array(processed_images)
    images_np = images_np.transpose((0, 3, 1, 2))
    tensor = torch.tensor(images_np)
    return tensor
