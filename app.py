
import torch
import torch.nn as nn
import json
import io

from PIL import Image
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile


# =========================
# 1. Load class names
# =========================

with open("class_names.json", "r") as f:
    data = json.load(f)

class_names = data["class_names"]


# =========================
# 2. Load model
# =========================

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(class_names)
)

model.load_state_dict(
    torch.load(
        "resnet18_plant_disease.pth",
        map_location="cpu"
    )
)

model.eval()


# =========================
# 3. Image preprocessing
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# 4. Prediction function
# =========================

def predict_image(image):

    image = image.convert("RGB")

    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_batch)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted_idx = torch.max(
        probabilities,
        dim=1
    )

    predicted_idx = predicted_idx.item()
    confidence = confidence.item()

    predicted_class = class_names[predicted_idx]

    return {
        "class": predicted_class,
        "confidence": round(confidence * 100, 2)
    }


# =========================
# 5. FastAPI
# =========================

app = FastAPI(
    title="Plant Disease Detection API",
    description="API for detecting plant diseases using ResNet18"
)


@app.get("/")
def home():

    return {
        "message": "Plant Disease Detection API is running!"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    result = predict_image(image)

    return result


print("app.py created successfully!")
