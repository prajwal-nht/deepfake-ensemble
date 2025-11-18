# models/model2.py
import torch
from transformers import ViTForImageClassification, ViTConfig
from safetensors.torch import load_file as safe_load
import json

CONFIG_FILE_PATH = "preprocess/preprocessor_config2.json"
def load_model(device, model_filename, config_path=CONFIG_FILE_PATH):
    """
    Loads the ViT-based deepfake detection model for Model2.
    
    Args:
        device (torch.device): The device to load the model onto.
        model_filename (str): Name of the weight file (e.g. model2.safetensors).
        config_path (str): Path to the JSON config file.
    
    Returns:
        model (torch.nn.Module): The loaded model.
    """
    # Load configuration settings from config.json
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = ViTConfig.from_dict(config_dict)
    model = ViTForImageClassification(config)
    
    # Set label mapping (using the config information)
    # Note: config.json keys "id2label" and "label2id" are strings; we convert keys to int.
    model.config.id2label = {int(k): v for k, v in config_dict.get("id2label", {}).items()}
    model.config.label2id = config_dict.get("label2id", {})
    
    # Load pretrained weights from the safe file.
    model_path = model_filename
    if model_filename.endswith(".safetensors"):
        state_dict = safe_load(model_path)
        model.load_state_dict(state_dict)
    else:
        # Fallback to torch.load if needed.
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    
    # pytorch_model.bin is the typical Hugging Face weight file but if you have .safetensors, you don't need it for prediction.
    model.to(device)
    model.eval()
    return model

def run_inference(model, input_tensor, device):
    """
    Runs a forward pass on the input tensor and returns the average probability 
    for the "Fake" class.
    
    Args:
        model (torch.nn.Module): The loaded ViT model.
        input_tensor (torch.Tensor): Preprocessed images tensor of shape (N, C, H, W).
        device (torch.device): Device for inference.
    
    Returns:
        float: Average probability for the fake label.
    """
    with torch.no_grad():
        outputs = model(input_tensor.to(device))
        logits = outputs.logits  # shape: (batch_size, num_labels)
        probs = torch.softmax(logits, dim=-1)
        # Assuming index 1 is the "Fake" class.
        fake_prob = probs[:, 1].mean().item()
    return fake_prob
