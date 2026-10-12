
from pathlib import Path
import torch
from app.model import MODEL, WEIGHTS

Path("models").mkdir(exist_ok=True)

torch.save(
    {
        "architecture": "ResNet50",
        "weights": "ImageNet-1K V2",
        "state_dict": MODEL.cpu().state_dict(),
        "classes": WEIGHTS.meta["categories"]
    },
    "models/resnet50.pth"
)

print("Model saved to models/resnet50.pth")
