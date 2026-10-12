
from pathlib import Path
from datetime import date

from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles

from app.model import MODEL, CLASSES, DEVICE, predict_image
from app.utils import read_and_validate_image

app = FastAPI(
    title="Image Classification API",
    description="Image classification using pretrained ResNet50",
    version="1.0.0"
)


@app.get("/api/health")
def health():
    return {"status": "healthy"}


@app.get("/api/classes")
def get_classes():
    return {
        "total_classes": len(CLASSES),
        "classes": CLASSES
    }


@app.get("/api/model/info")
def model_info():
    return {
        "architecture": "ResNet50",
        "weights": "ImageNet-1K V2",
        "number_of_classes": len(CLASSES),
        "training_date": None,
        "accuracy": {
            "top1_percent": None,
            "top5_percent": None
        },
        "device": str(DEVICE),
        "note": "Measure accuracy using evaluation/evaluate.py."
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    image = await read_and_validate_image(file)
    return predict_image(image)


frontend_path = Path("frontend")

if frontend_path.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(frontend_path), html=True),
        name="frontend"
    )
