import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


# -------------------------
# Paths & Device
# -------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "resnet18_ham10000_finetuned.pt"
LABELS_PATH = BASE_DIR / "model" / "labels.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Load labels
# -------------------------
with open(LABELS_PATH, "r") as f:
    LABELS: List[str] = json.load(f)

label_to_idx = {name: i for i, name in enumerate(LABELS)}
mel_idx = label_to_idx.get("melanoma", None)


# -------------------------
# Model
# -------------------------
model = models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, len(LABELS))

state = torch.load(MODEL_PATH, map_location="cpu")
if isinstance(state, dict) and "model_state" in state:
    model.load_state_dict(state["model_state"])
else:
    model.load_state_dict(state)

model.eval()
model.to(device)


# -------------------------
# Transforms
# -------------------------
IMG_SIZE = 224
tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Skin AI")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# -------------------------
# Utils
# -------------------------
def triage_advice(top1_label: str, melanoma_prob: float | None):
    urgent = {"melanoma"}
    moderate = {"basal_cell_carcinoma", "actinic_keratoses"}

    if top1_label in urgent or (melanoma_prob is not None and melanoma_prob >= 0.5):
        return (
            "This looks potentially serious. Please see a dermatologist as soon as possible, "
            "especially if the lesion is new, changing, bleeding, or painful. "
            "(This is not a diagnosis.)"
        )
    if top1_label in moderate:
        return (
            "Consider booking a dermatology appointment soon. "
            "If it changes quickly or bleeds, seek care earlier. "
            "(This is not a diagnosis.)"
        )
    return (
        "This may be benign. Monitor for changes in size, color, border, or bleeding. "
        "If you are concerned, consult a dermatologist. "
        "(This is not a diagnosis.)"
    )


@torch.no_grad()
def predict_image(img: Image.Image, topk: int = 3):
    x = tfms(img.convert("RGB")).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    top_probs, top_idxs = torch.topk(probs, k=min(topk, probs.numel()))
    top3 = [
        {"label": LABELS[int(i)], "prob": float(p)}
        for p, i in zip(top_probs, top_idxs)
    ]

    melanoma_prob = float(probs[mel_idx]) if mel_idx is not None else None
    advice = triage_advice(top3[0]["label"], melanoma_prob)

    return top3, melanoma_prob, advice


# -------------------------
# Routes
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "num_classes": len(LABELS),
        "labels": LABELS,
        "made_by": "Eng. Kirolles Ezzat"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file)
    top3, melanoma_prob, advice = predict_image(img)

    return {
        "top3": top3,
        "melanoma_probability": melanoma_prob,
        "advice": advice
    }
