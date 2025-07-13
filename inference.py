import os
import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
import torchvision.transforms as transforms
import cv2
import numpy as np

# ========== Configuration ==========
MODEL_PATH = "weights/model_fold_0.pt"  # Change this to your desired model
IMAGE_PATH = "test_spectrograms/sample.png"  # Path to input PNG spectrogram
TARGET_SHAPE = (224, 224)  # VGG16 input size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {0: 'normal', 1: 'murmur'}

# ========== Transform ==========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                         std=[0.229, 0.224, 0.225])
])

# ========== Load & Preprocess Image ==========
def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TARGET_SHAPE, interpolation=cv2.INTER_AREA)
    img = transform(img).unsqueeze(0)  # Shape: [1, 3, 224, 224]
    return img.to(DEVICE)

# ========== Load Model ==========
def load_model(model_path):
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    # Freeze first 5 conv layers
    conv_count = 0
    for layer in model.features:
        if isinstance(layer, nn.Conv2d):
            conv_count += 1
            if conv_count <= 5:
                for param in layer.parameters():
                    param.requires_grad = False
    model.classifier[6] = nn.Linear(4096, 2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ========== Inference ==========
def predict(image_path, model_path):
    model = load_model(model_path)
    img_tensor = load_image(image_path)

    with torch.no_grad():
        output = model(img_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()
        label = LABEL_MAP[pred_class]

    return label, confidence

# ========== Run ==========
if __name__ == "__main__":
    label, conf = predict(IMAGE_PATH, MODEL_PATH)
    print(f"Prediction: {label} (Confidence: {conf:.4f})")
