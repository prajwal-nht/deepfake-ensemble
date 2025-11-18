# model1.py
import torch
from transformers import ViTForImageClassification
from safetensors.torch import load_file as safe_load

def load_model(device, model_filename):
    """
    Loads the pretrained ViT-based deepfake detection model.
    
    Args:
        device (torch.device): The device to load the model onto.
        model_filename (str): Filename for the model weights (e.g., model1.safetensors, model1.pth, model1.pt).
    
    Returns:
        model (torch.nn.Module): The loaded model in evaluation mode.
    """
    # Specify the base model string used for training.
    model_str = "google/vit-base-patch16-224-in21k"
    num_labels = 2  # e.g. "Real" and "Fake"
    model = ViTForImageClassification.from_pretrained(model_str, num_labels=num_labels)
    model.config.id2label = {0: "Real", 1: "Fake"}
    model.config.label2id = {"Real": 0, "Fake": 1}

    # Construct the full file path (assuming the model is in the pretrained_models folder)
    model_path = model_filename
    if model_filename.endswith(".safetensors"):
        state_dict = safe_load(model_path)
        model.load_state_dict(state_dict)
    else:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model

def run_inference(model, input_tensor, device):
    """
    Runs a forward pass on the input tensor and returns the average probability for the "Fake" class.
    
    Args:
        model (torch.nn.Module): The loaded ViT model.
        input_tensor (torch.Tensor): Preprocessed image tensor of shape (N, C, H, W).
        device (torch.device): The device used for inference.
    
    Returns:
        float: The average probability (confidence) for the "Fake" class across the batch.
    """
    with torch.no_grad():
        outputs = model(input_tensor.to(device))
        # outputs.logits shape: (batch_size, num_labels)
        logits = outputs.logits
        # Apply softmax to get probabilities.
        probs = torch.softmax(logits, dim=-1)
        # Assume class index 1 corresponds to "Fake" (deepfake)
        deepfake_prob = (1 - probs[:, 1]).mean().item()
    return deepfake_prob
