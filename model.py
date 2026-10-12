
import time

import torch
from torchvision.models import resnet50, ResNet50_Weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHTS = ResNet50_Weights.DEFAULT
CLASSES = WEIGHTS.meta["categories"]
PREPROCESS = WEIGHTS.transforms()

MODEL = resnet50(weights=WEIGHTS).to(DEVICE)
MODEL.eval()


@torch.inference_mode()
def predict_image(image):
    start = time.perf_counter()

    tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)
    logits = MODEL(tensor)
    probabilities = torch.softmax(logits, dim=1)[0]

    scores, indices = torch.topk(probabilities, k=5)

    top5 = [
        {
            "class_name": CLASSES[index.item()],
            "confidence": round(score.item() * 100, 2)
        }
        for score, index in zip(scores, indices)
    ]

    return {
        "predicted_class": top5[0]["class_name"],
        "confidence": top5[0]["confidence"],
        "top5": top5,
        "inference_time_seconds": round(
            time.perf_counter() - start, 4
        )
    }
