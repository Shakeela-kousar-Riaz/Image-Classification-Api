
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from app.model import MODEL, DEVICE, PREPROCESS, CLASSES

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "test_data"
OUTPUT_DIR = ROOT
OUTPUT_DIR.mkdir(exist_ok=True)

if not DATA_DIR.exists():
    raise FileNotFoundError(
        "Create evaluation/test_data with class-named subfolders first."
    )

# Use only folders whose names match ImageNet labels.
dataset = ImageFolder(DATA_DIR, transform=PREPROCESS)

class_to_index = {name: i for i, name in enumerate(CLASSES)}

if not dataset.classes:
    raise ValueError("No class folders found in the test dataset.")

unknown = [name for name in dataset.classes if name not in class_to_index]
if unknown:
    raise ValueError(
        f"Folder names must match ImageNet labels exactly: {unknown}"
    )

loader = DataLoader(dataset, batch_size=16, shuffle=False)

true_labels = []
pred_labels = []
top5_correct = 0
total = 0

MODEL.eval()

with torch.inference_mode():
    for images, folder_labels in loader:
        images = images.to(DEVICE)
        logits = MODEL(images)
        top5 = torch.topk(logits, k=5, dim=1).indices.cpu()

        for row, folder_label in zip(top5, folder_labels):
            true_index = class_to_index[
                dataset.classes[folder_label.item()]
            ]
            true_labels.append(true_index)
            pred_labels.append(row[0].item())
            top5_correct += int(true_index in row.tolist())
            total += 1

top1 = accuracy_score(true_labels, pred_labels) * 100
top5_accuracy = top5_correct / total * 100

print(f"Test images: {total}")
print(f"Top-1 accuracy: {top1:.2f}%")
print(f"Top-5 accuracy: {top5_accuracy:.2f}%")

labels_present = sorted(set(true_labels))
matrix = confusion_matrix(
    true_labels,
    pred_labels,
    labels=labels_present
)

names = [CLASSES[index] for index in labels_present]
display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=names
)
display.plot(
    xticks_rotation=90,
    include_values=len(names) <= 15
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close()

report = {
    "architecture": "ResNet50",
    "test_images": total,
    "top1_percent": round(top1, 2),
    "top5_percent": round(top5_accuracy, 2)
}

with open(OUTPUT_DIR / "metrics.json", "w") as file:
    json.dump(report, file, indent=2)

print("Saved evaluation/confusion_matrix.png")
print("Saved evaluation/metrics.json")
