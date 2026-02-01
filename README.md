# Skin-Disease-Classification-using-Deep-Learning


# Skin Disease Classification (Deep Learning)

End-to-end deep learning project for skin lesion classification using the HAM10000 dataset.

## Features
- ResNet18 trained on 7 skin disease classes
- Handles severe class imbalance with class weighting
- Top-3 predictions with probabilities
- Melanoma probability output
- Simple triage message (educational only)

## Tech Stack
- PyTorch, Torchvision
- FastAPI
- HTML / CSS / JavaScript
- AWS SageMaker (CPU-only, cost-optimized)

## Dataset
- HAM10000 (not included in this repository)

## Run locally
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
