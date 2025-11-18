import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

def preprocess_images(image_paths):
    size = 224  
    transform = A.Compose([
        A.Resize(height=size, width=size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    processed_images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Image not loaded: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transformed = transform(image=img)
        processed_images.append(transformed['image'])
    return torch.stack(processed_images)